import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

import time 

def get_significant_features(X, y):
    model = ExtraTreesRegressor()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    
    #feat_importances.nlargest(10).plot(kind='barh') # horitonzal bar plot
    #plt.show()
    
    #print(feat_importances.nlargest(10))
    
    features = list(feat_importances.nlargest(10).keys())
    return features

def predict_points(model, player_info):
    
    points = 0
    
    # get the average over 100 predictions
    for i in range(100):
        points = points + model.predict(player_info)[0]
        
    return points/100
    
def get_percent_diff(a, b):
    diff = a - b
    avg = (a + b)/2
    
    if avg == 0:
        return None
    
    percent_diff = (diff/avg) * 100
    
    return percent_diff

def get_data(file):
    data = pd.read_csv(file)
    x = data.iloc[:,4:27] # independent columns with player name, team, and position removed to keep only numeric features 
    y = data.iloc[:,-1]   # target FantasyPoints columns
    
    return x, y
    
def predict_set(model, x, y):
    diff_percent = []
    player_points = []
    for i in range(len(x)):
        player = [x[i].tolist()]
        points = predict_points(model, player)
        player_points.append(points) # creating a list of predicted players points to be used as a sample output 
        percent = get_percent_diff(points, y[i])
        
        if percent != None:
            diff_percent.append(percent)
    
    print("Running data set through model......")
    print("Printing sample prediction for 5 players...............")
    
    for i in range(5):
        print("PLAYER " + str(i+1) + " Predicted: " + str(player_points[i]) + " Actual: " + str(y[i]))
        
    print("Total Average Percent Difference: " + str(mean(diff_percent)))

    
if __name__ == "__main__":
    start_time = time.time()

    # get dev data
    dev_X , dev_y = get_data("data/player_dev_set.csv")
    
    # get train data
    train_X, train_y = get_data("data/player_train_set.csv")
    
    # get 2019 test data
    x_2019, y_2019 = get_data("data/2019.csv")
    
    # reconfigure dataset to only feature the top 10 most significant features in determining fantasy score of a player
    significant_features = get_significant_features(train_X, train_y)
    train_X = train_X[significant_features]
    dev_X = dev_X[significant_features]
    x_2019 = x_2019[significant_features]

    # convert data to be compatible with Random Forest libraries
    train_X = train_X.to_numpy()
    dev_X = dev_X.to_numpy()
    x_2019 = x_2019.to_numpy()
    
    #print(dev_X[2])
    #print(dev_y[2])
    
    train_y = train_y.to_numpy()

    
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    
    # evalute model using repeated kfold cross validation 
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    X, y = make_regression(n_samples=10000, n_features=10, n_informative=15, noise=0.1, random_state=2)
    
    # report mean absolute error (MAE) for model
    # the larger the negative MAE the better; perfect model has a MAE of 0 
    #scores = cross_val_score(model, train_X, train_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    

    #print("--- Training Run ---")
    print("MAE: " + str(mean(scores)))
    print("Standard Deviation: " + str(std(scores)))
  
    '''
    print("--- Single Test ---")
    row = [[2, 12, 16, 72, 72, 581, 581, 165, 165, 0]]
    points = predict_points(model, row)
    diff = get_percent_diff(points, dev_y[2])
    print('Data Input: ' + str(row))
    print('Prediction: ' + str(points))
    print('Actual Points: ' + str(dev_y[2]))
    print('Percentage Diff: ' + str(diff) + "%")
    '''
    

    # attempt to predict fantasy points from dev data set
    #print("--- Fantasy Point Prediction for Dev Set ---")
    #predict_set(model, dev_X, dev_y)
    
    print("")
    
    # attempt to predict fantasy points from 2019 data set
    print("--- Fantasy Point Prediction for 2019 Data Set ---")
    #predict_set(model, x_2019, y_2019)
    
    print("Execution time: %s seconds " % (time.time() - start_time))
    
