from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler







#Pipeline for Train Data
class Train_Pipeline():

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
    




#Pipeline for Test data
class Validation_Pipeline():
    
    def __init__(self):
        pass

    def Convert_Categorical_To_Numerical_Features(self,validation_data):
        self.validation_data = validation_data
        
        try:
            categorical_variables = self.validation_data.columns[self.validation_data.dtypes=='object']
            target_variable = self.validation_data.iloc[:,-1].name
            
            #Number of folds for CV
            n_splits = 3
            kf = KFold(n_splits=n_splits,shuffle=True)
            
            #Looping through variable
            for variable in categorical_variables:
                    
                    #Create a new column for K-Fold Target Encoding
                    self.validation_data[f"KFold_Target_Encoding_{variable}"] = 0.0

                    #Loop through each fold
                    for fold, (train_indx,val_indx) in enumerate(kf.split(self.validation_data)):
                        train_fold = self.validation_data.iloc[train_indx]
                        val_fold = self.validation_data.iloc[val_indx]

                        #Calculate the mean of the target variable for each category in the training fold
                        encoding_map = train_fold.groupby(variable)[target_variable].mean().to_dict()

                        #Apply the encoding to the validation code
                        self.validation_data.loc[val_indx,f"KFold_Target_Encoding_{variable}"] = val_fold[variable].map(encoding_map)

            yield self.validation_data
            
        except:
             print("Check the data is in right format. Check the data")




    def Encoding_Values(self,validation_data):
         
         self.validation_data = validation_data
         
         categorical_variables = self.validation_data.columns[self.validation_data.dtypes=='object']
         
         self.encoders_map = []
         for variable in categorical_variables:
              
              #Calculate the mean of the for each categorical variable in the test data
              encoding_values = self.validation_data.groupby(variable)[f"KFold_Target_Encoding_{variable}"].mean().to_dict()
              encoding_values = encoding_values

              self.encoders_map.append(encoding_values)
              
              yield encoding_values


         
    def Remove_Categorical_Variables(self,validation_data):
        
        self.validation_data = validation_data
        
        categorical_variables = self.validation_data.columns[self.validation_data.dtypes=='object']
        
        for variable in categorical_variables:
            
            self.validation_data.drop(variable,axis=1,inplace=True)

            yield self.validation_data

    

    def Validation_XY(self,validation_data):

        self.validation_data = validation_data
    
        xvalidation = self.validation_data.drop('Churn',axis=1)
        yvalidation = self.validation_data['Churn']
        
        return (xvalidation,yvalidation)  
    
    
    #Normalization
    def validation_Normalization(self,xvalidation):
         
        self.xvalidation = xvalidation
         
        std = StandardScaler()
        xvalidation_norm = std.fit_transform(self.xvalidation)
         
        return xvalidation_norm
    


