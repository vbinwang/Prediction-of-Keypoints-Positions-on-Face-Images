#!/usr/bin/env python
'''Functions and classes for partitioning the dataset.
'''
import cPickle as pickle
import os

import numpy as np


class Partitioner(object):
    '''Partitions a Dataset into classes based on the label-size dictionary
    '''

    def __init__(self, dataset, labels_sizes, cache_file, shuffle=True, ):
        '''Creates a partitioner class.

        Args:
            datatset: A dictionary containing an X/Y that can be partitioned.
            labels-sizes: a dictionary of labels-to-sizes (in 0-100%), note
                all sizes must add up to 100, this is enforced
            shuffle: an optional variable that shuffles the data-set before
                partitioning them.

        '''
        self.__dataset = dataset
        assert sum(labels_sizes.values()) == 100
        self.__labels_sizes = labels_sizes
        self.__cache_file = cache_file
        self.__shuffle = shuffle

        self.__labels_indices = None

    @staticmethod
    def __generate_partition_indices(num_samples, labels_sizes, shuffle):
        # Create a list of the indices to use when slicing the dataset and
        # optionally shuffle it.
        indices = range(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        partition = {}
        start_idx = 0
        samples_used = 0
        for key in labels_sizes.keys():
            # use floating-point precision to gobble up all of the samples.
            samples_used += float(labels_sizes[key])/100. * num_samples
            end_idx = int(np.round(samples_used))
            partition[key] = indices[start_idx: end_idx]

            start_idx = end_idx
        print "%d/%d" % (start_idx, num_samples)
        assert start_idx == num_samples

        return partition

    def __save_indices(self):
        '''Saves the partition indices to disk for later retrieval.
        '''
        with open(self.__cache_file, 'wb') as file_descriptor:
            pickler = pickle.Pickler(file_descriptor)
            pickler.dump(self.__labels_indices)

    def __load_indices(self):
        '''Loads the partition indices from disk to recreate state.
        '''
        with open(self.__cache_file, 'rb') as file_descriptor:
            unpickler = pickle.Unpickler(file_descriptor)
            self.__labels_indices = unpickler.load()

    def run(self):
        '''
        Returns:
            A dict containing named partitioned data returned in a dict.  For
                example:

            {'train': {'X': <dataset.Xs>, 'Y': <dataset.Ys>},
             'test': {'X': <dataset.Xs>, 'Y': <dataset.Ys>},
             'validate': {'X': <dataset.Xs>, 'Y': <dataset.Ys>}}

        '''
        if os.path.exists(self.__cache_file):
            print "Using partitioning index cache file %s" % self.__cache_file
            self.__load_indices()
        else:
            self.__labels_indices = Partitioner.__generate_partition_indices(
                len(self.__dataset['X']), self.__labels_sizes, self.__shuffle)
            self.__save_indices()

        partitioned = {}
        for k in self.__labels_indices:
            partitioned[k] = {
                'X': self.__dataset['X'][self.__labels_indices[k]],
                'Y': self.__dataset['Y'][self.__labels_indices[k]],
                'Index': self.__labels_indices[k],
                'Y_Labels': self.__dataset['Y_Labels']
            }
        return partitioned
