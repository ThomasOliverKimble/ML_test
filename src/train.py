import csv
from helpers import *
from preprocessing import *
from models import *
from config import *
from data_augmented import *


# Load train data 
print('Loading train data')
data = load_csv_data(DATA_TRAIN_PATH)

# Data cleaning 
print('Cleaning data')
preprocessed_data_train = preprocess_data(data)

# This mode permits to validate our models
if MODE == 'validation' : 
    # Variables to calculate overall score on validation set
    score_tot = 0
    len_tot = 0
    
    for cat in JET_CATEGORIES: 
        # Pre-process data
        y_train = preprocessed_data_train[cat][:,1]
        x_train = preprocessed_data_train[cat][:,2:]
        
        # Print score for each category 
        print("Scores for category:", cat)
        score_test, len_test = model_validation(y_train, x_train)
        score_tot += score_test*len_test
        len_tot +=  len_test

    # Calculate total score
    total_score = score_tot/len_tot
    print('Total testing score percentage: ' + str(round(total_score*100,3)) + '%')

# this mode permits to train our chosen model and save the weights 
elif MODE == 'train': 
    weights = []
    for cat in JET_CATEGORIES: 
        # Pre-process data
        y_train = preprocessed_data_train[cat][:,1]
        x_train = preprocessed_data_train[cat][:,2:]

        # Procede to data augmentation
        x_train_augmented = feature_generation(x_train, BEST_DEGREES_CAT[cat])

        # Apply model to our augmented data
        w,_ = model(y_train, x_train_augmented, cat)

        # Save weights
        with open(W_PATH + '_'+ str(cat)+ '.csv',  'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(w)
            f.close()

