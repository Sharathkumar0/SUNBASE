
def Encoding_Map_Values(Test_data):
    
    
    categorical_variables = Test_data.columns[Test_data.dtypes=='object']

    for variable in categorical_variables:

        #Calculate the mean of the for each categorical variable in the test data
        encoding_values = Test_data.groupby(variable)[f"KFold_Target_Encoding_{variable}"].mean().to_dict()

        yield encoding_values

