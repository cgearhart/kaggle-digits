#!/usr/bin/env python
# coding: utf-8
"""
Script to predict values using a pkl model file and data stored
in a numpy array file and store the results in a csv file. (Modified
version of pylearn2 predict_csv.py)

Basic usage:

.. code-block:: none

    predict_conv.py model_pkl.pkl data.npy output.csv

"""
from __future__ import print_function

import sys
import argparse
import numpy as np
import pandas as pd

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

__authors__ = ["Zygmunt Zając", "Marco De Nadai", "Chris Gearhart"]
__license__ = "GPL"


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch a prediction from a pkl file"
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('test_filename',
                        help='Specifies the file with the values to predict')
    parser.add_argument('output_filename',
                        help='Specifies the output file')
    return parser


def predict(model_path, test_path, output_path):
    """
    Predict using a model from a pkl file.

    Parameters
    ----------
    modelFilename : str
        The file name of the model file.
    testFilename : str
        The file name of the file to test/predict.
    outputFilename : str
        The file name of the output file.
    """

    print("Loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("Error loading {}:".format(model_path))
        print(e)
        return False

    print("Setting up symbolic expressions...")
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y = T.argmax(Y, axis=1)

    f = function([X], Y, allow_input_downcast=True)

    print("Loading data and predicting...")
    x = np.load(open(test_path, 'rb'))  # x is a numpy.ndarray
    m, _, _, _ = x.shape
    step = m / 100  # TODO:: make batch size a parameter

    print("Making predictions...")
    y = np.ndarray(m, dtype=np.int)
    for i in range(0, m, step):
        y[i:i+step+1] = f(x[i:i+step+1, :, :, :]).astype(np.int)
    final_predictions = pd.Series(y, name="Label")
    final_predictions.index += 1  # adjust the index to start at 1
    final_predictions.to_csv(output_path, index_label="ImageId", header=True)

    print("Done.")

    return True


if __name__ == "__main__":

    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename,
                  args.test_filename,
                  args.output_filename
                  )
    if not ret:
        sys.exit(-1)
