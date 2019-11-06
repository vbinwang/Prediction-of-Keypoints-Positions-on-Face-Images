#!/usr/bin/env python
'''Data Logging classes
'''
import abc
import os
import shutil

import numpy as np
import pandas as pd


class EpochLogger(object):
    '''Logs data for each epoch
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def log(self, data, epoch):
        '''Logs the given data for the given epoch
        '''
        pass


class CSVEpochLogger(EpochLogger):
    '''Logs data for each epoch using a csv file
    '''
    EPOCH_COLNAME = "epoch"

    def __init__(self, file_fmt_str, link_file, column_names):
        super(CSVEpochLogger, self).__init__()
        self.__file_fmt = file_fmt_str
        self.__link_file = os.path.join(link_file)
        self.__column_names = np.concatenate(
            ([self.EPOCH_COLNAME], column_names))

    def _get_filename(self, epoch):
        return self.__file_fmt % epoch

    def log(self, data, epoch):
        data_frame = pd.DataFrame([np.concatenate(([epoch], data))],
                                  columns=self.__column_names)
        data_frame[[self.EPOCH_COLNAME]] = (
            data_frame[[self.EPOCH_COLNAME]].astype(int))
        filename = self._get_filename(epoch)
        if epoch == 1:
            # Create a new file
            with open(filename, 'w') as file_desc:
                data_frame.to_csv(file_desc, index=False)
        else:
            shutil.copy2(self._get_filename(epoch-1), filename)
            with open(filename, 'a') as file_desc:
                data_frame.to_csv(file_desc, index=False, header=False)

        # Update the symlink
        if (os.path.exists(self.__link_file) and
                os.path.islink(self.__link_file)):
            os.unlink(self.__link_file)
        os.symlink(filename, self.__link_file)
