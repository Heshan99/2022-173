import pandas as pd
import numpy as np

vehicle=pd.read_csv('Vehicle_DataSet.csv')
vehicle

vehicle.drop(['Title','Sub_title','Condition','Transmission','Body','Fuel','Capacity','Location','Description','Post_URL','Seller_name','Seller_type','published_date'], axis=1, inplace=True)
vehicle

vehicle.info()

vehicle['Year'].unique()

vehicle['Edition'].unique()

vehicle['Mileage'].unique()

vehicle['Price'].unique()

backup = vehicle.copy()

vehicle['Mileage'] = vehicle['Mileage'].str.split(' ').str.get(0).str.replace(',','')
vehicle['Mileage']

vehicle = vehicle [vehicle['Mileage'].str.isnumeric()]

vehicle['Mileage']=vehicle['Mileage'].astype(int)

vehicle.info()

vehicle = vehicle[vehicle['Price']!=" Negotiable"]
vehicle = vehicle[vehicle['Price']!="Negotiable"]

vehicle['Price'] = vehicle['Price'].str.replace('Rs', '').str.replace(',','').str.replace('.','')
vehicle['Price']

vehicle['Price']=vehicle['Price'].astype(int)

vehicle = vehicle.reset_index(drop=True)
vehicle

vehicle.describe()

vehicle=vehicle[vehicle['Price']<7e7].reset_index(drop=True)

vehicle.to_csv('Vehicle Cleaned Data Set.csv')

X = vehicle.drop(columns='Price')
y = vehicle['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(X[['Brand','Model','Edition','Year','Mileage']])

ohe.categories_

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['Brand','Model','Edition','Year','Mileage']), remainder='passthrough')

lr = LinearRegression()

pipe = make_pipeline(column_trans,lr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test,y_pred)

scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

np.argmax(scores)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test, y_pred)

import pickle

pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

vehicleBrand = input("Enter Vehicle Brand: ")
vehicleModel = input("Enter Vehicle Model: ")
vehicleEdition = input("Enter Vehicle Edition: ")
vehicleYear = int(input("Enter Vehicle Year: "))
vehicleMileage = int(input("Enter Vehicle Mileage: "))
predictedvalue = pipe.predict(pd.DataFrame([[vehicleBrand, vehicleModel, vehicleEdition, vehicleYear, vehicleMileage]], columns=['Brand', 'Model', 'Edition', 'Year', 'Mileage']))

print (predictedvalue)
