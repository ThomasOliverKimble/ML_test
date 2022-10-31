import numpy as np
from implementations import *
from helpers import *
from config import *
from preprocessing import *
from data_augmented import *


def get_accuracy(y,label):
    """ 
    Get accuracy.
    Args : 
        y: prediction
        label: true value 
    Return: 
        acc: accuracy
    """
    acc = np.sum(label == y)/len(y)
    return acc

def build_k_indices(y, k_fold, seed=0):
    """build k indices for k-fold.
    Args:
        y: prediction, shape=(N,)
        k_fold: fold num
        seed: the random seed
    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)

def cross_validation_acc(y, x, lambdas, degrees, k_fold):
    """Return the accuracy 
    Args:
        y: prediction, shape=(N,)
        x: data
        lambdas : lambdas to test
        degrees: degrees to test
        k_fold: fold num
    Returns:
        acc: accuracy
    """
    train_fold = []
    test_fold = []
    folds = []
    k_indices = build_k_indices(y, k_fold, seed=1)
    features_x = np.arange(2,x.shape[1])
    
    for k in range(k_fold):
        test = k_indices[k]
        others = np.delete(k_indices,k)
        if k == 0:
            train = others
        else:
            train = np.concatenate((train,others))
        
        folds.append((train,test))
        
    acc = np.zeros((len(DEGREES),len(LAMBDAS)))
    
    for d, deg in enumerate(degrees):
        print(deg)
        poly_x = poly_generation(x,features_x, deg)
        for l, lmb in enumerate(lambdas):
            k_acc = []
            for train_ind,test_ind in folds:
                train_x,train_y = poly_x[train_ind],y[train_ind]
                test_x,test_y = poly_x[test_ind],y[test_ind]
                w_ridge,_ = ridge_regression(train_y, train_x, lmb)
                
                label = np.sign(test_x @ w_ridge)*1.0
                accuracy = get_accuracy(test_y,label)
                k_acc = np.append(k_acc,accuracy)
            acc[d][l] = np.mean(k_acc)
    return acc

def cross_validation():
    """Performs cross validation and save results for all combination of lambdas and degrees."""
    # Load data 
    data = load_csv_data(DATA_TRAIN_PATH)

    # Data cleaning 
    print('Cleaning data')
    preprocessed_data_train = preprocess_data(data)

    for cat in JET_CATEGORIES: 
        print("Do cross validation")
        y_train = preprocessed_data_train[cat][:,1]
        x_train = preprocessed_data_train[cat][:,2:]
        accuracy_array = cross_validation_acc(y_train, x_train, LAMBDAS, DEGREES, k_fold=10)
        np.save((CROSS_ARRAY_PATH + '_'+ str(cat)), accuracy_array)

# To perform cross validation when one's launch this file       
cross_validation()