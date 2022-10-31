from helpers import *
from preprocessing import *
from config import *
from data_augmented import *

# Load test data 
print('Loading test data')
data = load_csv_data(DATA_TEST_PATH)

# Data cleaning 
print('Pre-process data test')
preprocessed_data_test = preprocess_data(data)


# Testing our data with model predifined
print('Testing starts')
results = []
for cat in JET_CATEGORIES:
    
    # Load weights
    weights= np.genfromtxt(W_PATH + '_'+ str(cat)+ '.csv', delimiter=",")
    
    print('Perform data augmentation')
    # Augment testing data and save  
    # If you want to do it again, you can de uncomment the next 3 lines and comment the 4th
    x_test =  preprocessed_data_test[cat][:,2:]
    x_test_augmented = feature_generation(x_test, BEST_DEGREES_CAT[cat])
    np.save((AUGMENTED_TEST_PATH + '_'+ str(cat)), x_test_augmented)
    # x_test_augmented = np.load((AUGMENTED_TEST_PATH + '_'+ str(cat))


    # Get predicted values for testing set
    yt_test = np.where(x_test_augmented@weights.T > 0.0, 1, -1)
    
    # Save results for all category with corresponding Ids 
    ids_extend = np.expand_dims(preprocessed_data_test[cat][:,0], axis=1)
    ytest_extend = np.expand_dims(yt_test, axis=1)
    results_cat = np.concatenate((ids_extend, ytest_extend), axis=1)
    
    if cat == 0:
        results = results_cat
    else:
        results = np.concatenate((results, results_cat), axis=0)

# Sort by Ids
results = results[results[:, 0].argsort()]
    
#Create csv file for submission
print('Creating file submission')
create_csv_submission(results[:,0], results[:,1], SUBMISSION_PATH)

