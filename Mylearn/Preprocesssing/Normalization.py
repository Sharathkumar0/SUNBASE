
from sklearn.preprocessing import StandardScaler

#Normalization
def normalization(xtrain,xtest):
    
    std = StandardScaler()
    
    return std.fit_transform(xtrain),std.fit_transform(xtest)