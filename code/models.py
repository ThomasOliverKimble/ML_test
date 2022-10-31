# -*- coding: utf-8 -*-
import numpy as np
from helpers import *
from implementations import *
from config import *

def model_validation(y, x):
    """
    This function separate training set in 80% training set and 20% validation set then it applies the method
    chosen.
    Print all the scores of our validation set
    Args:
        y: predictions data
        x: training set
    Returns:
        score_test: score for training set
        len(y_train): lenght of dataset
    """
    # Split data for validation
    x_train, x_test, y_train, y_test = split_data(x, y, 0.8)
    
    # Compute number of features
    number_features = x_train.shape[1]
    # Take initial_w with only zeros
    initial_w = np.zeros(number_features)
    
    # Get w from method
    if METHOD=='Least_square':    
        w, loss = least_squares(y_train, x_train)
    elif METHOD=='Gradient_descent':
        w, loss = mean_squared_error_gd(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Stochastic_gradient_descent':
        w, loss = mean_squared_error_sgd(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Ridge_regression':
         w, loss = ridge_regression(y_train, x_train, LAMBDA)
    elif METHOD=='Logistic_regression':
        w, loss = logistic_regression(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Reg_logistic_regression':
        w, loss = reg_logistic_regression(y_train, x_train, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    
    # Get predicted values
    yt_train = np.where(x_train@w.T > 0.0, 1, -1)
    yt_test = np.where(x_test@w.T > 0.0, 1, -1)
    
    # Get scores
    score_train = (yt_train == y_train).sum()/len(y_train)
    score_test = (yt_test == y_test).sum()/len(y_test)
    print('Training score percentage: ' + str(round(score_train*100,3)) + '%')
    print('Testing score percentage: ' + str(round(score_test*100,3)) + '%')
    
    return score_test, len(y_train)

def model(y_train, x_train, cat):
    """
    Train the model chosen and return the weights 
    Args:
        y_train: predictions train data
        x_train: training set
        cat: category to know which hyperparameters to use
    Returns:
        w: weights computed
        loss: loss computed
    """ 
    # Compute number of features
    number_features = x_train.shape[1]
    # Take initial_w with only zeros
    initial_w = np.zeros(number_features)
    
    # Train our chosen model
    if METHOD=='Least_square':
        w, loss = least_squares(y_train, x_train)
    elif METHOD=='Gradient_descent':
        w, loss = mean_squared_error_gd(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Stochastic_gradient_descent':
        w, loss = mean_squared_error_sgd(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Ridge_regression':
         w, loss = ridge_regression(y_train, x_train, BEST_LAMBDAS_CAT[cat])
    elif METHOD=='Logistic_regression':
        w, loss = logistic_regression(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    elif METHOD=='Reg_logistic_regression':
        w, loss = reg_logistic_regression(y_train, x_train, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    
    return w, loss
