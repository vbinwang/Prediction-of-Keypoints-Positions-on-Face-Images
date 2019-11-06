#!/usr/bin/env python
'''This module defines a multi-level perceptron and convolutional MLP.
'''
import abc
import json

import lasagne
import theano
import theano.tensor as T

import adaptive_updates


class MultiLevelPerceptron:
    '''Base Class for the Multi-level Perceptron (defines the interface)

    All MLP-like classes should inherit from this one to be compatible with
    the batch-processing classes.
    '''
    PREDICT_MISSING = 'predict_missing'

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def build_network(self):
        '''Compiles the network.
        '''
        pass

    @abc.abstractmethod
    def train(self, x_values, y_values):
        '''Updates the model and calculates RMSE loss for the given X, Y data.
        '''
        pass

    @abc.abstractmethod
    def validate(self, x_values, y_values):
        '''Returns the RMSE loss for the given model and a set of X, Y data.
        '''
        pass

    @abc.abstractmethod
    def predict(self, x_values):
        '''Predicts Y from the model and the given X data.
        '''
        pass

    @abc.abstractmethod
    def get_state(self):
        '''Get the state of the model and return it.
        '''
        pass

    @abc.abstractmethod
    def set_state(self, state):
        '''Set the state of the model from the argument.
        '''
        pass

    @abc.abstractmethod
    def epoch_done_tasks(self, epoch, num_epochs):
        '''Call this at the end of each epoch to perform parameter updates.
        '''
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class ConvolutionalMLP(MultiLevelPerceptron):
    '''Convolutional MLP Definition.
    '''
    __LINEARITY_TYPES = {
        'rectify': lasagne.nonlinearities.rectify,
        'tanh': lasagne.nonlinearities.tanh,
        'leaky_rectify': lasagne.nonlinearities.leaky_rectify,
        'sigmoid': lasagne.nonlinearities.sigmoid
    }

    # This isn't great, but it's a one-off
    def __init__(self, config, input_shape, output_width):
        '''Creates a Convolutional Multi-level MultiLevel

        '''
        super(ConvolutionalMLP, self).__init__()
        if config[MultiLevelPerceptron.PREDICT_MISSING]:
            if 'output_nonlinearity' in config:
                print "Overriding Output %s to Sigmoid for Binary Prediction"
            else:
                print "Using Output Sigmoid for Binary Prediction"
            config['output_nonlinearity'] = "sigmoid"

        self.__config = config
        self.__input_shape = (config['batchsize'],) + input_shape
        self.__output_width = output_width

        self.__input_var = T.tensor4('input')
        self.__target_var = T.matrix('target')

        self._network = None
        self._create_network()
        self.__train_fn = None
        self.__validate_fn = None
        self.__adaptive_updates = []

    def _create_network(self):
        if self._network is not None:
            raise AssertionError('Cannot call BuildNetwork more than once')

        # pylint: disable=redefined-variable-type
        nonlinearity = self.__LINEARITY_TYPES[self.__config['nonlinearity']]

        # Input Layer
        lyr = lasagne.layers.InputLayer(self.__input_shape, self.__input_var,
                                        name='input')
        if 'input_drop_rate' in self.__config:
            lyr = lasagne.layers.DropoutLayer(
                lyr,
                p=self.__config['input_drop_rate'],
                name='input_dropout')

        # 2d Convolutional Layers
        if 'conv' in self.__config:
            i = 0
            for conv in self.__config['conv']:
                lyr = lasagne.layers.Conv2DLayer(
                    lyr,
                    num_filters=conv['filter_count'],
                    filter_size=tuple(conv['filter_size']),
                    nonlinearity=nonlinearity,
                    name=('conv_2d_%d' % i))
                if 'pooling_size' in conv:
                    lyr = lasagne.layers.MaxPool2DLayer(
                        lyr,
                        pool_size=tuple(conv['pooling_size']),
                        name=('pool_2d_%d' % i))
                if 'dropout' in conv and conv['dropout'] != 0:
                    lyr = lasagne.layers.DropoutLayer(
                        lyr,
                        p=conv['dropout'],
                        name=('conv_dropout_%d' % i))
                i += 1

        # Hidden Layers
        if 'hidden' in self.__config:
            i = 0
            for hidden in self.__config['hidden']:
                lyr = lasagne.layers.DenseLayer(
                    lyr,
                    num_units=hidden['width'],
                    nonlinearity=nonlinearity,
                    W=lasagne.init.GlorotUniform(),
                    name=('dense_%d' % i))

                if 'dropout' in hidden and hidden['dropout'] != 0:
                    lyr = lasagne.layers.DropoutLayer(
                        lyr,
                        p=hidden['dropout'],
                        name=('dropout_%d' % i))
                i += 1

        if 'output_nonlinearity' in self.__config:
            output_nonlinearity = self.__LINEARITY_TYPES[
                self.__config['output_nonlinearity']]
        else:
            output_nonlinearity = None

        # Output Layer
        self._network = lasagne.layers.DenseLayer(
            lyr, num_units=self.__output_width,
            nonlinearity=output_nonlinearity,
            name='output')

    def epoch_done_tasks(self, epoch, num_epochs):
        for updater in self.__adaptive_updates:
            updater.update(epoch, num_epochs)

    def build_network(self):
        # The output of the entire network is the prediction, define loss to be
        # the RMSE of the predicted values + optional l1/l2 penalties.
        prediction = lasagne.layers.get_output(self._network)
        if self.__config[MultiLevelPerceptron.PREDICT_MISSING]:
            print "Selecting binary cross-entropy loss"
            objective = lasagne.objectives.binary_crossentropy
            loss = objective(prediction, self.__target_var).mean() * 4192.
        else:
            print "Selecting squared-error loss"
            objective = lasagne.objectives.squared_error
            loss = objective(prediction, self.__target_var).mean()

        if 'l1' in self.__config and self.__config['l1']:
            print "Enabling L1 Regularization"
            loss += lasagne.regularization.regularize_network_params(
                self._network, lasagne.regularization.l1)

        if 'l2' in self.__config and self.__config['l2']:
            print "Enabling L2 Regularization"
            loss += lasagne.regularization.regularize_network_params(
                self._network, lasagne.regularization.l2) * 1e-4

        # Training accuracy is simply the rmse.
        train_prediction = lasagne.layers.get_output(
            self._network, deterministic=True)
        training_accuracy = objective(
            train_prediction, self.__target_var)
        training_accuracy = lasagne.objectives.aggregate(
            training_accuracy, mode='mean')

        # Grab the parameters and define the update scheme.
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        shared_learning_rate, learning_rate_updater = (
            adaptive_updates.adaptive_update_factory(
                'learning_rate', self.__config))
        if learning_rate_updater is not None:
            self.__adaptive_updates.append(learning_rate_updater)

        shared_momentum, momentum_updater = (
            adaptive_updates.adaptive_update_factory(
                'momentum', self.__config))
        if momentum_updater is not None:
            self.__adaptive_updates.append(momentum_updater)

        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=shared_learning_rate,
            momentum=shared_momentum)

        # For testing the output, use the deterministic parts of the output
        # (this turns off noise-sources, if we had any and possibly does things
        # related to dropout layers, etc.).  Again, loss is defined using rmse.
        test_prediction = lasagne.layers.get_output(
            self._network, deterministic=True)
        test_loss = objective(
            test_prediction, self.__target_var)

        # Create the training and validation functions that we'll use to train
        # the model and validate the results.
        self.__train_fn = theano.function(
            [self.__input_var, self.__target_var],
            [loss, training_accuracy], updates=updates)
        self.__validate_fn = theano.function(
            [self.__input_var, self.__target_var],
            [test_loss, test_loss])

    def predict(self, x_values):
        return(lasagne.layers.get_output(
            self._network, x_values, deterministic=True).eval())

    def train(self, x_values, y_values):
        return self.__train_fn(x_values, y_values)

    def validate(self, x_values, y_values):
        return self.__validate_fn(x_values, y_values)

    def get_state(self):
        return lasagne.layers.get_all_param_values(self._network)

    def set_state(self, state):
        lasagne.layers.set_all_param_values(self._network, state)

    def __str__(self):
        ret_string = "Convoluational MLP:\n%s\n" % (
            json.dumps(self.__config, sort_keys=True))

        lyrs = lasagne.layers.get_all_layers(self._network)
        ret_string += "  Layer Shapes:\n"
        for lyr in lyrs:
            ret_string += "\t%20s = %s\n" % (
                lyr.name, lasagne.layers.get_output_shape(lyr))
        return ret_string


class AmputatedMLP(ConvolutionalMLP):

    def __init__(self, config, input_shape, output_width):
        super(AmputatedMLP, self).__init__(
            config, input_shape, output_width)

    def predict(self, x_values):
        lyrs = lasagne.layers.get_all_layers(self._network)
        return(lasagne.layers.get_output(
            lyrs[len(lyrs)-2], x_values, deterministic=True).eval())
