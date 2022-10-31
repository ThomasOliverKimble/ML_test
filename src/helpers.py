# -*- coding: utf-8 -*-
import csv
import numpy as np


def load_csv_data(path_dataset, sub_sample=False):
    """
    Loads data in csv format
    Args:
        path_dataset: path of data to load.
        sub_sample: allows to sub-sample data (50 samples).
    Returns:
        data: numpy array containing the dataset.
    """
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)

    #ids (event ids) are integer values
    data[:,0] = data[:, 0].astype(np.int)

    #y (class labels) are converted : (b,s,?) in (-1,1,1)
    y = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1], converters={1: lambda x: -1 if b"b" in x else 1})
    data[:,1] = y

    #sub_sample : 50 samples
    if sub_sample :
        data = data[::50] 
            
    return data


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd (Provided function)
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Args :
        y:  output, shape=(N, 1)
        tx: data sample, hape=(N, D)
        batch_size: size of batches
        num_batches: batch number
        shuffle: allows to shuffle shuffle data if True
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. 
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """                             
    nb_samples= x.shape[0]
    random_index = np.random.permutation(nb_samples)
    nb_samples_ratio = int(np.floor(ratio*nb_samples))
    
    x_mixed = x[random_index]
    x_tr = x_mixed[:nb_samples_ratio]
    x_te = x_mixed[nb_samples_ratio:]
    
    y_mixed = y[random_index]
    y_tr = y_mixed[:nb_samples_ratio]
    y_te = y_mixed[nb_samples_ratio:]
    
    return x_tr, x_te, y_tr, y_te


