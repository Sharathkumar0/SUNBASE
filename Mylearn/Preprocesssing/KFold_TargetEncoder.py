from sklearn.model_selection import KFold

def Convert_Categorical_To_Numerical_Features(Train_data,Test_data):

  
    for conversion_data in [Train_data,Test_data]:
        try:
            categorical_variables = conversion_data.columns[conversion_data.dtypes=='object']
            target_variable = conversion_data.iloc[:,-1].name

            #Number of folds for CV
            n_splits = 3
            kf = KFold(n_splits=n_splits,shuffle=True)


            #Looping through variable
            for variable in categorical_variables:

                #Create a new column for K-Fold Target Encoding
                conversion_data[f"KFold_Target_Encoding_{variable}"] = 0.0

                #Loop through each fold
                for fold, (train_indx,val_indx) in enumerate(kf.split(conversion_data)):
                    train_fold = conversion_data.iloc[train_indx]
                    val_fold = conversion_data.iloc[val_indx]

                    #Calculate the mean of the target variable for each category in the training fold
                    encoding_map = train_fold.groupby(variable)[target_variable].mean().to_dict()

                    #Apply the encoding to the validation code
                    conversion_data.loc[val_indx,f"KFold_Target_Encoding_{variable}"] = val_fold[variable].map(encoding_map)


            yield conversion_data
   
        except:
            print("Check the data is in right format. Check the data")


    
        