from sklearn.tree import DecisionTreeClassifier

def Model(xtrain,ytrain):
    
    model = DecisionTreeClassifier()
    model.fit(xtrain,ytrain)
    
    return model