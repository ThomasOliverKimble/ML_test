# -*- coding: utf-8 -*-
"""some data_cleaning functions."""
import numpy as np
from helpers import *
from config import *


def divide_data(data):
    """
    Divide data depending on column categories 
    Args:
        data: data to separate
    Returns:
        cat_array: numpy array containing the divided dataset.
    """
    cat_array = []

    #Check every line in the dataset to separate data wrt category
    for cat in JET_CATEGORIES : 
        tx_train_cat = data[np.where(data[:,JET_CATEGORIES_INDEX]==cat)]
        tx_train_cat = np.delete(tx_train_cat, JET_CATEGORIES_INDEX, 1)
        
        cat_array.append(tx_train_cat)

    return np.array(cat_array, dtype=object)


def clean_data(data):
    """
    Clean data : Replace outliers value by Nan, replace all Nan value by the median, mean, std or variance
        if train/validation mode : remove and save columns with more than 90% of Nan values and columns containing the same value (std = 0),                        
        if test mode : delete the columns that were defined during training mode
    Args:
        data: raw data 
    Returns:
        data_cat: numpy array containing the cleaned dataset.
    """
    data_cat = []

    for cat in JET_CATEGORIES : 
        #Replace undefined data (-999) by NaN
        data[cat][data[cat] <= np.float64(UNDEF)] = np.nan 

        # If we train/validate, we select the columns to remove, delete and save them
        if(MODE != 'test'):
            #Select and remove columns with a percentage of NaN (90%) and with a std equal to 0 
            col_to_delete_nans = np.where(np.divide(np.sum(np.isnan(data[cat][:,2::]),axis=0),np.shape(data[cat])[0]) > 0.9)
            col_to_delete_std = np.where(np.nanstd(data[cat][:,2::],axis=0) == 0) 
            col_to_delete = np.concatenate((col_to_delete_nans[0], col_to_delete_std[0]), axis=0)  
            data[cat] = np.delete(data[cat],col_to_delete+2,axis=1) 
            
            # Save columns number to delete 
            with open(COL_PATH + '_'+ str(cat)+ '.csv',  'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(col_to_delete)
                f.close()

        # If we test, we delete the columns that were deleted in the training set
        elif(MODE == 'test'):
            col_to_del= np.genfromtxt(COL_PATH + '_'+ str(cat)+ '.csv', delimiter=",")
            if col_to_del.shape != ():
                col_to_del = col_to_del.astype(np.int)
                data[cat] = np.delete(data[cat],col_to_del+2,axis=1) 

        # Find and replace outliers by NaN
        data[cat] = find_outliers(data[cat])

        #Replace Nan values by the 'replace' value : median/mean/std/var
        if (METH_REPLACEMENT == 'median'):
            col_med = np.nanmedian(data[cat], axis=0)
            for i, col in enumerate(col_med):
                data[cat][:,i] = np.nan_to_num(data[cat][:,i], nan=col_med[i])
        if(METH_REPLACEMENT == 'mean'):
            col_mean = np.nanmean(data[cat], axis=0)
            for i, col in enumerate(col_mean):
                data[cat][:,i] = np.nan_to_num(data[cat][:,i], nan=col_mean[i])
        if(METH_REPLACEMENT == 'std'):
            col_std = np.nanstd(data[cat], axis=0)
            for i, col in enumerate(col_std):
                data[cat][:,i] = np.nan_to_num(data[cat][:,i], nan=col_std[i])
        if(METH_REPLACEMENT == 'var'):
            col_var = np.nanstd(data[cat], axis=0)
            for i, col in enumerate(col_var):
                data[cat][:,i] = np.nan_to_num(data[cat][:,i], nan=col_var[i])   

        # Standardize data 
        data[cat][:,2::] = standardize(data[cat][:,2::])    
            
        data_cat.append(data[cat])

    return np.array(data_cat, dtype=object)

def find_outliers(data):
    """
    Replace outliers by Nan value. If the value is smaller than 25% or bigger than 75% of the data it is an outlier.
    Args:
        data: data 
    Returns:
        data: numpy array containing data with Nan values instead of outliers.
    """
    #column=features
    col_to_del = []
    for i in range(data.shape[1]):
        column = data[:,i]
        p25 = np.percentile(column, 25)
        p75 = np.percentile(column, 75)
        inter = p75-p25
        
        upper_lim = p75+(1.5*inter)
        lower_lim = p25-(1.5*inter)
        
        #Nan if outlier, zero if not
        column = np.where(column>upper_lim, np.nan, column)
        column = np.where(column<lower_lim, np.nan, column)
        data[:,i] = column

    return data

def standardize(x):
    """
    Standardize the data set x
    Args:
        x: data to standardize
    Returns:
        x: the standardize data
    """
    #mean for every column
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def preprocess_data(data):
    """
    Pre-process the data by dividing it into 4 categories depending on PRI_JET_NUM column (24th) then cleaning the data and 
    finally normalizing it.
    Args:
        data: data to preprocess
    Returns:
        data_std: numpy array containing the preprocessed dataset.
    """
    # Divide data depending on PRI_jet_num value
    data_divide = divide_data(data)
    
    # Remove columns with 90% of undefined value (-999.0) and outliers
    data_clean = clean_data(data_divide)
    
    return data_clean
