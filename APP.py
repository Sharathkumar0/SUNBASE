# importing Flask and other modules
from flask import Flask, request, render_template
import pandas as pd
import pickle

import requests
import pandas as pd
import base64


#Load the model
with open('Model.pkl','rb') as f:
    Model = pickle.load(f)


# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
        customer_id = request.form.get("fcustomerid")
        customer_name = request.form.get("fcustomername")
        customer_age = request.form.get("fcustomerage")
        customer_sub_plan = request.form.get("fcustomerplan")
        customer_bill = request.form.get("fcustomerbill")
        customer_usage = request.form.get("fcustomerusage")
        customer_gender = request.form.get("fcustomergender")
        customer_location = request.form.get("fcustomerlocation")

        l1 = ['CustomerID', 'Name', 'Age', 'Gender', 'Location','Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
        l2 = [customer_id,customer_name,customer_age,customer_gender.title(),customer_location.title(),customer_sub_plan, customer_bill,customer_usage]

        i = dict(zip(l1,l2))
        
        Test_Data = pd.DataFrame()
        Test_Data = Test_Data.append(i,ignore_index=True)        

        Encoding_Gender = {'Female': 0.4966372126491822, 'Male': 0.5009816139202232}
        Encoding_Location = {'Chicago': 0.5023632271000893,'Houston': 0.4932437180240546,'Los Angeles': 0.4880537354475553,
                             'Miami': 0.5057768054012094,'New York': 0.5049853126124199}


        Test_Data.drop(['CustomerID','Name'],axis=1,inplace=True)

        Categorical_Features = Test_Data.columns[Test_Data.dtypes == "object"]
        for variable in Categorical_Features:
            if variable == 'Gender':
                Test_Data.loc[Test_Data.index,'Gender'] = Test_Data['Gender'].map(Encoding_Gender)
            elif variable == 'Location':
                Test_Data.loc[Test_Data.index,'Location'] = Test_Data['Location'].map(Encoding_Location)
        
        Test_Data.rename(columns={'Gender':'KFold_Target_Encoding_Gender','Location':'KFold_Target_Encoding_Location'},inplace=True)

        Test_Data = Test_Data.loc[:,['Age','Subscription_Length_Months','Monthly_Bill',
                             'Total_Usage_GB','KFold_Target_Encoding_Gender','KFold_Target_Encoding_Location']]
        
        #Predictions
        ypredictions = Model.predict(Test_Data)
        
        if ypredictions[0] == 1:
            res = "Customer is going to stop the subscription plan"
        else:
            res = "Customer is enjoying the subscription plan and going to continue"

    return render_template('index.html',result=res)

if __name__ == "__main__":
     app.run(debug=True)
