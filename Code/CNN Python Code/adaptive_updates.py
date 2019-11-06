#!/usr/bin/env python
'''Adaptive update classes to automagically update the learning rate and
momentum after each epoch.
'''
import abc

import numpy as np
import theano


class EndOfEpochUpdate(object):
    '''Base-class for a class that runs at the end of each epoch of training
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, epoch, total_epochs):
        '''Updates the shared value based at the end of the epoch.
        '''
        pass


class AdaptiveUpdate(EndOfEpochUpdate):
    '''Moves in log-linear or linear steps after each epoch.
    '''

    def __init__(self, param_name, update_type, param_vals, shared_var):
        super(AdaptiveUpdate, self).__init__()

        self.__init_fn = {
            "log": self.__log_init,
            "linear": self.__lin_init
        }[update_type]
        self.__init_done = False

        self.__param_name = param_name
        self.__param_vals = param_vals
        self.__shared_var = shared_var
        self.__values = None

    def __log_init(self, total_epochs):
        self.__values = np.logspace(
            np.log10(self.__param_vals['start']),
            np.log10(self.__param_vals['end']),
            num=total_epochs)

    def __lin_init(self, total_epochs):
        self.__values = np.linspace(
            self.__param_vals['start'],
            self.__param_vals['end'],
            num=total_epochs)

    def update(self, epoch, total_epochs):
        if not self.__init_done:
            self.__init_fn(total_epochs)
            self.__init_done = True

        self.__shared_var.set_value(np.float32(self.__values[epoch]))
        print "Setting %s to %3.3e" % (
            self.__param_name, self.__shared_var.get_value())


def adaptive_update_factory(param, config):
    '''Returns a class that updates after each epoch (or None if not adaptive)

    Args:
        param (str): base-name of the parameter
        config: dictionary of configuration values

    Returns:
        theano.shared: the shared parameter to give to the model
        EndOfEpochUpdate or None: returns a updater class or None if a
            static parameter was found.
    '''
    param_prefixes = {
        "log": "log_",
        "linear": ""
    }
    param_suffixes = {
        "start": "_start",
        "end": "_end"
    }

    def create_shared(initial_value):
        '''Creates a shared theano value set to the initial value.
        '''
        return theano.shared(np.float32(initial_value), name=param)

    for update_type, prefix in param_prefixes.iteritems():
        required_vars = {var_fn: prefix + param + suffix for (var_fn, suffix)
                         in param_suffixes.items()}
        if all(var in config.keys() for var in required_vars.values()):
            values = {var_fn: config[var_name] for (var_fn, var_name)
                      in required_vars.items()}
            print ("Selecting adaptive %s update for %s with initial "
                   "value=%3.2e" % (update_type, param, values['start']))
            shared = create_shared(values['start'])
            return shared, AdaptiveUpdate(param, update_type, values, shared)

    initial_value = config[param]
    print "Using static %s with value=%3.2e" % (
        param, initial_value)
    shared = create_shared(initial_value)
    return shared, None
