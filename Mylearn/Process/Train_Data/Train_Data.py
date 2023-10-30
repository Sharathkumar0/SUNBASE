from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler







#Pipeline for Train Data
class Process_Train_Data():

    def __init__(self):
        pass

    def Convert_Categorical_To_Numerical_Features(self,Train_data):

        self.Train_data = Train_data
        
        try:
            categorical_variables = self.Train_data.columns[self.Train_data.dtypes=='object']
            target_variable = self.Train_data.iloc[:,-1].name
            
            #Number of folds for CV
            n_splits = 3
            kf = KFold(n_splits=n_splits,shuffle=True)
            
            #Looping through variable
            for variable in categorical_variables:
                    #Create a new column for K-Fold Target Encoding
                    self.Train_data[f"KFold_Target_Encoding_{variable}"] = 0.0

                    #Loop through each fold
                    for fold, (train_indx,val_indx) in enumerate(kf.split(self.Train_data)):
                        train_fold = self.Train_data.iloc[train_indx]
                        val_fold = self.Train_data.iloc[val_indx]

                        #Calculate the mean of the target variable for each category in the training fold
                        encoding_map = train_fold.groupby(variable)[target_variable].mean().to_dict()

                        #Apply the encoding to the validation code
                        self.Train_data.loc[val_indx,f"KFold_Target_Encoding_{variable}"] = val_fold[variable].map(encoding_map)

                    self.Train_data.drop(variable,axis=1,inplace=True)


            yield self.Train_data
            
        except:
             print("Check the data is in right format. Check the data")



    def Train_XY(self,Train_data):

        self.Train_data = Train_data
    
        xtrain = self.Train_data.drop('Churn',axis=1)
        ytrain = self.Train_data['Churn']
        
        return (xtrain,ytrain)  
    
    
    #Normalization
    def Train_Normalization(self,xtrain):
         
        self.xtrain = xtrain
         
        std = StandardScaler()
        xtrain_norm = std.fit_transform(self.xtrain)
         
        return xtrain_norm