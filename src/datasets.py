"""
datasets loading module
Author: Emmanuel Larralde
"""

import os
import gzip
import pickle

import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) #Directory of this file
ROOT_PATH = os.path.dirname(DIR_PATH) #Directory of the project


class DataSet:
    """
    DataSet class: Contains input and targets for training, validation and testing.
    """
    def __init__(self, train, val, test) -> None:
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:
        """
        String representation of the class. Displays the number of items per set.
        """
        to_format =  "DataSet(train: {}, test: {}, val: {})"
        return to_format.format(
            len(self.train[0]),
            len(self.test[0]),
            len(self.val[0])
        )


def load_digits() -> tuple:
    """
    Loads provided mnist dataset
    """
    with gzip.open(f'{ROOT_PATH}/data/mnist.pkl.gz','rb') as ff :
        u = pickle._Unpickler( ff )
        u.encoding = 'latin1'
        train, val, test = u.load()
    return DataSet(train, val, test)
