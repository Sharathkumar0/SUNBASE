
#Import required libraries
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold



def Train_Test_Split(data='None',splits=4):
    
    try:
    
        #Intializing KFold CV
        num_splits = splits
        kf = KFold(n_splits=num_splits,shuffle=True)


        #Getting indexes for train and test
        slice_index = random.randint(0,(len(list(kf.split(data)))-1))
        train_indx,test_indx = list(kf.split(data))[slice_index]

        #Creating dataframe for Train data
        Train_data = pd.DataFrame()
        Train_data = pd.concat([Train_data,data.iloc[train_indx]])
        Train_data.index = range(len(Train_data))

        #Creating dataframe for Test data
        Test_data = pd.DataFrame()
        Test_data = pd.concat([Test_data,data.iloc[test_indx]])
        Test_data.index = range(len(Test_data))
        
        return Train_data,Test_data
    
    except:
        print('Please give the required data or Check the input data format')