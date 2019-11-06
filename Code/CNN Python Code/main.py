#!/usr/bin/env python
'''Main Program for Training over the Facial Keypoints dataset.
'''
import argparse
import code
import datetime
import json
import os
import re
import subprocess
import sys
import time

import argcomplete
import lasagne
import numpy as np
import pandas as pd

import batch
import data_logger
import fileio
import partition
import perceptron
import preprocess

ACTION_TRAIN_MISSING = "train_missing"


class Tee(object):
    '''Tees file descriptors so that writes are made everywhere simultaneously.
    '''
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        '''Writes to all file descriptors.
        '''
        # pylint: disable=invalid-name
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        '''Flushes all file descriptors.
        '''
        # pylint: disable=invalid-name
        for f in self.files:
            f.flush()


def get_version():
    '''Returns the current git revision as a string
    '''
    return "git rev = %s" % (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())


def print_partition_shapes(partitions):
    for k in partitions.keys():
        assert len(partitions[k]['X']) == len(partitions[k]['Y'])
        assert len(partitions[k]['Index']) == len(partitions[k]['X'])
        assert (list(partitions[k]['Y'].shape)[1] ==
                len(partitions[k]['Y_Labels']))
        print("%8s X.shape=%s, Y.shape=%s, indices=%s, labels=%s" % (
            k, partitions[k]['X'].shape, partitions[k]['Y'].shape,
            len(partitions[k]['Index']), len(partitions[k]['Y_Labels'])))


def load_nnet_config(options):
    '''Load the neural network config file and address overrides
    '''
    print "Loading NNET configuration from %s" % options.config_file
    assert os.path.exists(options.config_file)
    with open(options.config_file) as json_fp:
        nnet_config = json.load(json_fp)

    # Handle command-line overrides that affect the config
    if options.batchsize is not None:
        nnet_config['batchsize'] = options.batchsize

    if options.action == ACTION_TRAIN_MISSING:
        nnet_config[perceptron.MultiLevelPerceptron.PREDICT_MISSING] = True
    else:
        nnet_config[perceptron.MultiLevelPerceptron.PREDICT_MISSING] = False

    return nnet_config


def select_data(options, feature_cols, partitioned, keep_all_data=False):
    #
    # Select the columns.
    #
    if options.action == ACTION_TRAIN_MISSING:
        binary = {}
        for name in partitioned.keys():
            num_vars = len(partitioned[name]['Y_Labels'])
            odd_stride = range(1, num_vars, 2)
            even_stride = range(0, num_vars, 2)

            nans = np.isnan(partitioned[name]['Y'])
            even_cols = nans[:, odd_stride]
            odd_cols = nans[:, even_stride]

            assert np.array_equal(odd_cols, even_cols)
            col_names = ([re.sub(r'_[xy]$', "", l) for
                         l in partitioned[name]['Y_Labels']])
            even_colnames = [col_names[i] for i in even_stride]
            odd_colnames = [col_names[i] for i in odd_stride]
            assert np.array_equal(even_colnames, odd_colnames)

            binary[name] = {
                'X': partitioned[name]['X'],
                'Y': even_cols,
                'Index': partitioned[name]['Index'],
                'Y_Labels': ['missing_' + c for c in even_colnames]
            }

        print "Binarized the Missing Values"
        return binary

    selected = {}
    for name in partitioned.keys():
        selected[name] = {}
        selected[name]['X'] = partitioned[name]['X']
        selected[name]['Y'] = partitioned[name]['Y'][:, feature_cols]
        selected[name]['Y_Labels'] = (
            [partitioned[name]['Y_Labels'][i] for i in feature_cols])
        selected[name]['Index'] = partitioned[name]['Index']

    if keep_all_data:
        return selected

    #
    # Map/Drop NaNs
    #
    if options.keep_nans:
        replaced = 0
        total = 0
        for name in partitioned.keys():
            to_replace = np.isnan(selected[name]['Y'])
            selected[name]['Y'][to_replace] = options.nan_cardinal

            replaced += np.sum(to_replace)
            total += to_replace.size

        print "Replaced NaNs with cardinal=%d [%3.1f%% of data]" % (
            options.nan_cardinal, float(replaced)/float(total)*100.)
        return selected
    else:
        dropped = 0
        total = 0
        for name in partitioned.keys():
            to_keep = ~(np.isnan(selected[name]['Y']).any(1))
            selected[name]['X'] = selected[name]['X'][to_keep]
            selected[name]['Y'] = selected[name]['Y'][to_keep]
            selected[name]['Index'] = (
                [selected[name]['Index'][i] for (i, keep)
                    in enumerate(to_keep) if keep])

            assert not np.isnan(selected[name]['Y']).any()

            dropped += sum(~to_keep)
            total += len(to_keep)
        print "Dropping samples with NaNs: {:3.1f}% dropped".format(
            float(dropped)/float(total)*100.)
        return selected


def combine_loss_main(options):
    with open(options.feature_group_file) as feat_fd:
        feature_groups = json.load(feat_fd)

    start_time = time.time()
    combine_loss(options, feature_groups)
    print "Combining CSVs took {:.3f}s".format(time.time() - start_time)


def combine_loss(options, feature_groups):
    assert os.getcwd() == options.run_data_path

    loss_dfs = {}

    for feature_idx, (feature_name, feature_cols) in enumerate(sorted(
            feature_groups.items())):
        if not os.path.exists(feature_name):
            print "Could not find directory for %s" % feature_name
            continue
        os.chdir(feature_name)

        with open('loss.csv') as loss_fd:
            df = pd.read_csv(loss_fd, index_col="epoch")

        df.rename(columns={
            "train_loss": "train_loss_" + feature_name,
            "train_rmse": "train_rmse_" + feature_name
        }, inplace=True)

        loss_dfs[feature_name] = df

        os.chdir(options.run_data_path)

    assert os.getcwd() == options.run_data_path

    aggregated_loss = pd.concat(loss_dfs.values(), axis=1)
    with open("loss.csv", 'w') as write_fd:
        aggregated_loss.to_csv(write_fd)


def train_feature(options, selected, nnet_config,
                  original_data, feature_cols, kaggle):
    #
    # Instantiate and Build the Convolutional Multi-Level Perceptron
    #
    start_time = time.time()
    if options.amputate:
        print "Chose an Amputated MLP"
        perceptron_type = perceptron.AmputatedMLP
    else:
        print "Chose an Convolutional MLP"
        perceptron_type = perceptron.ConvolutionalMLP
    mlp = perceptron_type(
        nnet_config, (1, 96, 96), len(selected['train']['Y_Labels']))
    print mlp
    mlp.build_network()
    print "Building Network Took {:.3f}s".format(time.time() - start_time)

    #
    # Finally, launch the training loop.
    #
    print "Starting training..."
    loss_log = data_logger.CSVEpochLogger(
        "loss_%05d.csv", "loss.csv",
        np.concatenate((['train_loss', 'train_rmse'],
                        selected['train']['Y_Labels'])))
    resumer = batch.TrainingResumer(
        mlp, "epochs_done.txt", "state_%05d.pkl.gz",
        options.save_state_interval)
    # if options.action == ACTION_TRAIN_MISSING:
    #     in_out_scale = None
    # else:
    #     in_out_scale = batch.Scaler(offset=1., scale=48.)

    trainer = batch.BatchedTrainer(mlp, nnet_config['batchsize'],
                                   selected, loss_log, resumer)
    trainer.train(options.num_epochs)

    if options.amputate or options.predict or options.kaggle:
        def write_pred(data, filename, header):
            data_frame = pd.DataFrame(data, columns=header)
            data_frame[['index']] = data_frame[['index']].astype(int)
            with open(filename, 'w') as file_desc:
                data_frame.to_csv(file_desc, index=False)

        start_time = time.time()
        print "Saving Predictions"

        prediction_header = np.concatenate(
            (['index'], selected['validate']['Y_Labels']))
        last_layer_header = None
        if not options.amputate:
            last_layer_header = prediction_header
        if options.predict:
            index_train = np.transpose([original_data['train']['Index']])
            index_valid = np.transpose([original_data['validate']['Index']])

            last_layer_train = np.concatenate(
                (index_train, trainer.predict_y(original_data['train']['X'])),
                axis=1)
            last_layer_val = np.concatenate(
                (index_valid, trainer.predict_y(original_data['validate']['X'])),
                axis=1)
            write_pred(last_layer_train, "last_layer_train.csv",
                       last_layer_header)
            write_pred(last_layer_val, "last_layer_val.csv",
                       last_layer_header)
            all_data = select_data(options, feature_cols, original_data, True)
            write_pred(np.concatenate(
                    (index_train, all_data['train']['Y']), axis=1),
                "y_train.csv", prediction_header)
            write_pred(np.concatenate(
                    (index_valid, all_data['validate']['Y']), axis=1),
                "y_validate.csv", prediction_header)

        if options.kaggle:
            kaggle_test = np.concatenate(
                (np.transpose([kaggle['Index']]),
                 trainer.predict_y(kaggle['X'])),
                axis=1)
            write_pred(kaggle_test, "kaggle.csv", prediction_header)

        print "  took {:.3f}s".format(time.time() - start_time)


def select_features(options):
    if options.action == ACTION_TRAIN_MISSING:
        return {'all_binary': range(30)}

    with open(options.feature_group_file) as feature_fd:
        feature_dict = json.load(feature_fd)

    if options.feature_groups is None:
        features_to_train = feature_dict
    else:
        if not all(k in feature_dict for k in options.feature_groups):
            raise KeyError(
                ("one or more of the following features cannot be found %s" %
                    options.feature_groups))
        features_to_train = dict((k, feature_dict[k]) for k in
                                 options.feature_groups if k in feature_dict)

    return features_to_train


def train_main(options):
    '''This loads the data, manipulates it and trains the model, it's the glue
    that does all of the work.
    '''
    # Load the nnet_config
    nnet_config = load_nnet_config(options)

    #
    # Load the Dataset
    #
    start_time = time.time()
    faces = fileio.FaceReader(options.faces_csv, options.faces_pickled,
                              fast_nrows=options.num_rows)
    faces.load_file()
    print "Read Took {:.3f}s".format(time.time() - start_time)

    kaggle = None
    if options.kaggle:
        kaggle = fileio.ReadTestCSV(
            "../../data/test.csv", "../../data/test.pkl")

    #
    # Partition the Dataset
    #
    # Convert raw data from float64 to floatX
    # (32/64-bit depending on GPU/CPU)
    typed_data = faces.get_data()
    typed_data['X'] = lasagne.utils.floatX(typed_data['X'])
    typed_data['Y'] = lasagne.utils.floatX(typed_data['Y'])

    for i in range(typed_data['X'].shape[0]):
        typed_data['X'][i][0] = np.swapaxes(typed_data['X'][i][0], 0, 1)

    start_time = time.time()
    partitioner = partition.Partitioner(
        typed_data, {'train': 90, 'validate': 10},
        os.path.join(options.run_data_path, "partition_indices.pkl"))
    partitions = partitioner.run()
    print_partition_shapes(partitions)
    print "Partition Took {:.3f}s".format(time.time() - start_time)

    #
    # Run any transformations on the training dataset here.
    #

    #
    # Which features to predict
    #
    features_to_train = select_features(options)
    for feature_index, (feature_name, feature_cols) in enumerate(sorted(
            features_to_train.items())):
        #
        # Setup environment for training a feature
        #
        print "\n\n%s\nFeature Set %s (%d of %d)\n%s" % (
            "#" * 80, feature_name, feature_index+1,
            len(features_to_train), "#" * 80)

        feature_path = os.path.abspath(feature_name)
        print "Changing to %s" % feature_path
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        os.chdir(feature_path)

        #
        # Select the Training data
        #
        # Select feature columns
        start_time = time.time()
        print "Selecting features %s" % feature_name
        selected = select_data(options, feature_cols, partitions)
        print selected['train']['Y_Labels']
        print_partition_shapes(selected)
        print "Selecting Data Took {:.3f}s".format(time.time() - start_time)

        train_feature(options, selected, nnet_config,
                      partitions, feature_cols, kaggle)

        #
        # Change back to the run directory for the next run.
        #
        print "Changing to %s" % options.run_data_path
        os.chdir(options.run_data_path)

        # Drop into a console so that we do anything additional we need.
        if options.drop_to_console:
            code.interact(local=locals())

    # Finally combine all of the loss-functions to produce a loss output.
    start_time = time.time()
    combine_loss(options, features_to_train)
    print "Combining CSVs took {:.3f}s".format(time.time() - start_time)


def main():
    '''Parse Arguments and call real_main
    '''
    #
    # Create list of options and parse them.
    #
    parser = argparse.ArgumentParser(
        version=get_version(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'action', nargs='?', choices=('train', 'loss', ACTION_TRAIN_MISSING),
        default='train', help="action to perform")
    parser.add_argument(
        '-o', '--output_dir', dest='run_data_path',
        metavar="PATH",
        default=datetime.datetime.now().strftime('run_%Y-%m-%d__%H_%M_%S'),
        help="directory to place run information and state")
    parser.add_argument(
        '-c', '--config_file', dest='config_file',
        metavar="FILE", default="configs/default.cfg",
        help="neural network configuration file")
    parser.add_argument(
        '--console', dest='drop_to_console', action="store_true",
        help="drop to console after finishing processing")
    parser.add_argument(
        '--feature_groups', dest='feature_groups', metavar="FEAT",
        default=None, nargs="+",
        help="feature groups to train")
    parser.add_argument(
        '--amputate', dest='amputate', action="store_true",
        help="train a neural network and save output of penultimate layer")
    parser.add_argument(
        '--predict', dest='predict', action="store_true",
        help="predict output after training")
    parser.add_argument(
        '--kaggle', dest='kaggle', action="store_true",
        help="load the test data and predict values for kaggle")

    train_group = parser.add_argument_group(
        "Training Control", "Options for Controlling Training")
    train_group.add_argument(
        '-e', '--epochs', dest='num_epochs', type=int, metavar="EPOCHS",
        default=100,
        help="number of epochs to train against")
    train_group.add_argument(
        '-i', '--interval', dest='save_state_interval', type=int,
        metavar="EPOCHS",
        default=10,
        help="how often (in epochs) to save internal model state")
    train_group.add_argument(
        '-b', '--batchsize', dest='batchsize', type=int, metavar="ROWS",
        default=None,
        help="override the batchsize specified in config_file")

    data_group = parser.add_argument_group(
        "Data Options", "Options for controlling the input data.")
    data_group.add_argument(
        '--faces_csv', dest='faces_csv',
        metavar="PATH", default=os.path.abspath("../data/training.csv"),
        help="path to the faces CSV file")
    data_group.add_argument(
        '--faces_pickle', dest='faces_pickled',
        metavar="PATH", default=os.path.abspath("../data/training.pkl.gz"),
        help="path to the faces pickle file")
    data_group.add_argument(
        '--feature_group_file', dest='feature_group_file',
        metavar="PATH", default=os.path.abspath("feature_groups.json"),
        help="path to the featuer groups")
    data_group.add_argument(
        '--num_rows', dest='num_rows', type=int, metavar="ROWS",
        default=None,
        help="override to specify number of first rows to use")
    data_group.add_argument(
        '--keep_nans', dest='keep_nans', action="store_true",
        help="option to drop target NaNs instead of mapping to cardinal ")
    data_group.add_argument(
        '--nan_cardinal', dest='nan_cardinal', type=int, metavar="VALUE",
        default=-1,
        help="cardinal value to use for target NaNs")

    argcomplete.autocomplete(parser)
    options = parser.parse_args()

    # Create a directory for our work.
    if os.path.exists(options.run_data_path):
        print "Using existing data directory at %s" % options.run_data_path
    else:
        print "Creating data directory at %s" % options.run_data_path
        os.mkdir(options.run_data_path)

    options.run_data_path = os.path.abspath(options.run_data_path)

    # pylint: disable=too-many-locals
    # Yes we have a lot of variables, but this is the main function.

    # Make the config file path absolute (so that a chdir won't affect it)
    options.config_file = os.path.abspath(options.config_file)

    #
    # Change to the run data directory
    #
    print "Changing to Run Directory=%s" % options.run_data_path
    os.chdir(options.run_data_path)

    # Tee the output to a logfile.
    console_fd = open('console_log.txt', 'a')
    sys.stdout = Tee(sys.stdout, console_fd)
    sys.stderr = Tee(sys.stderr, console_fd)
    print "Tee done; subsequent prints appear on terminal and console_log.txt"

    #
    # Log the current git revision/cmdline for reproducibility
    #
    print get_version()
    print "cmdline = '%s'" % " ".join(sys.argv)

    # Run the function for the specified action.
    {
        'train': train_main,
        'loss': combine_loss_main,
        ACTION_TRAIN_MISSING: train_main
    }[options.action](options)


if __name__ == "__main__":
    main()
