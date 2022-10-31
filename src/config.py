import numpy as np

#==============================================================================
# Paths
#==============================================================================
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
SUBMISSION_PATH = '../logfiles/sample-submission.csv'
W_PATH = '../logfiles/weights'
COL_PATH = '../logfiles/col'
CROSS_ARRAY_PATH = '../logfiles/cross_array'
AUGMENTED_TRAIN_PATH = '../logfiles/data_augmented'
AUGMENTED_TEST_PATH ='../logfiles/test_augmented'
#==============================================================================
# Train
#==============================================================================
MAX_ITERS=3000
GAMMA=0.1
LAMBDA=0.1

METHOD = 'Ridge_regression' 
METH_REPLACEMENT = 'median'
MODE = 'train' #train or test or validation
#==============================================================================
# Cross-validation
#==============================================================================
LAMBDAS = np.logspace(-10,-2,30)
DEGREES = np.arange(2,12)
# Thanks to cross-validation we found those values
BEST_DEGREES_CAT = [8,7,10,9]
BEST_LAMBDAS_CAT = [np.logspace(-10,-2,30)[19], np.logspace(-10,-2,30)[15], np.logspace(-10,-2,30)[19], 
                    np.logspace(-10,-2,30)[17]]
#==============================================================================
# Data
#==============================================================================
JET_CATEGORIES = np.arange(4)
JET_CATEGORIES_INDEX = 24
UNDEF = -999.0

