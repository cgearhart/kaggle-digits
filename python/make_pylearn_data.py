
from os import path

import numpy as np
import pandas as pd

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial


def preprocess(df):
    """
    Converts grayscale images into LeNet-5 format by embedding them
    in a larger image, inverting the color spectrum, and changing the
    scale to expedite learning by controlling the distribution via
    the mean and variance.

    Parameters
    ----------
    df : Pandas DataFrame
        dataframe containing flattened 28x28 images on each row

    Returns np.ndarray in [b, 0, 1, c] tensor format
    """
    R, C = 28, 28  # MNIST data image size
    R_L, C_L = 32, 32
    MAX_VAL = 255.0  # max value in uint8 image
    FLOOR = -0.1
    CEIL = 1.175
    m, _ = df.shape
    X = np.ndarray((m, R_L, C_L, 1), dtype=np.float32)
    for i, row in enumerate(df.as_matrix()):
        im = row.reshape((R, C)).astype(np.float32) / MAX_VAL
        pad_im = np.pad(im, (2,), 'constant')  # pad 2 pixels to each edge
        X[i, :, :, :] = CEIL - ((CEIL - FLOOR) * pad_im[:, :, np.newaxis])
    X = X.reshape([m, R_L, C_L, 1])  # convert to theano tensor shape
    return X


def makeDesignMatrix(df, cv_ratio=None, batch_size=None):
    """
    Use a dataframe of training data to create DenseDesignMatrix instances
    for pylearn2, optionally splitting the data into training and cross
    validation sets.

    Parameters
    ----------
    df : Pandas DataFrame
        dataframe containing flattened image instances on each row
    cv_ratio : float
        number on the range [0, 1] for the fraction of data that should
        be in the cross validation set
    batch_size : int
        number of instances used in training batches; an error is raised
        when the number of rows in df is not a multiple of the batch size.
        batch_size is ignored if cv_ratio is None or 0.

    Returns DenseDesignMatrix tuple containing [training_data, cv_data]
    """

    X = preprocess(df.iloc[:, 1:])
    Y = np.atleast_2d(df['label'].as_matrix().astype(np.uint8)).T

    m, _ = df.shape
    n = m
    cv = None

    if cv_ratio:

        if not batch_size:
            batch_size = 1
        elif m % batch_size != 0:
            raise(ValueError("Number of training instances is not a "
                             + "multiple of batch_size."))

        n = int(m * (1 - cv_ratio) / batch_size) * batch_size
        cv = DenseDesignMatrix(topo_view=X[n:, :, :, :],
                               y=Y[n:, :],
                               axes=['b', 0, 1, 'c'],
                               y_labels=10)

    x = DenseDesignMatrix(topo_view=X[:n, :, :, :],
                          y=Y[:n, :],
                          axes=['b', 0, 1, 'c'],
                          y_labels=10)

    return [x, cv]


if __name__ == "__main__":
    TMP_DIR = "tmp"
    DATA_DIR = "data"

    print("Building test dataset...")
    test_df = pd.read_csv(path.join(DATA_DIR, "test.csv"), delimiter=",")
    test_data = preprocess(test_df)
    np.save(path.join(TMP_DIR, "np_test.npy"), test_data)

    print("Building training & cross-validation design matrices...")
    train_df = pd.read_csv(path.join(DATA_DIR, "train.csv"), delimiter=",")
    [x, cv] = makeDesignMatrix(train_df, cv_ratio=0.2, batch_size=100)

    x.use_design_loc(path.join(TMP_DIR, 'train_design.npy'))
    serial.save(path.join(TMP_DIR, 'train.pkl'), x)

    if cv:
        cv.use_design_loc(path.join(TMP_DIR, 'cv.npy'))
        serial.save(path.join(TMP_DIR, 'cv.pkl'), cv)

    print("Done.")