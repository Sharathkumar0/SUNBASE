
def Remove_Categorical_Variables(Train_data,Test_data):

    for data in [Train_data,Test_data]:
        categorical_variables = data.columns[data.dtypes=='object']
        for variable in categorical_variables:
            data.drop(variable,axis=1,inplace=True)

        yield data

