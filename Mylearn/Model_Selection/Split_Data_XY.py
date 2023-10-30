#Separate the predictors and target variable

def XY_Train_Test(Train_data,Test_data):
    
        xtrain = Train_data.drop('Churn',axis=1)
        ytrain = Train_data['Churn']
        
        xtest = Test_data.drop('Churn',axis=1)
        ytest = Test_data['Churn']
        
        return (xtrain,xtest,ytrain,ytest)  