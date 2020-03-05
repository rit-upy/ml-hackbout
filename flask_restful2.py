#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:18:00 2020

@author: ritvik
"""

from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

dataset = pd.read_csv('/home/ritvik/Desktop/dataset_for_hackathon/wheat-2014-supervised.csv')
#axis=1 indicates the columns are dropped
dataset = dataset.drop(['CountyName','State','DayInSeason','Date','visibility','precipTypeIsOther','precipProbability','precipIntensity'],axis = 1)



scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaler_for_post_input = StandardScaler()

df_x = dataset.iloc[:,0:-1]
df_x = df_x.fillna(df_x.mean())
df_y = dataset.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size = 0.2)
scaler_x.fit(x_train)
y_train = np.reshape(y_train.values,(-1,1))
scaler_y.fit(y_train)
x_train = scaler_x.transform(x_train)
y_train = scaler_y.transform(y_train)
x_test = scaler_x.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
#predicted = regressor.predict(x_test)
#predicted = scaler_y.inverse_transform(predicted)
#y_test = y_test.values



app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        
        return {'about':'Hello World'}
    
    def post(self):
        some_json = request.get_json()
        print(dir(request))
        
        a = np.array([some_json['Latitude'],
                      some_json['Longitude'],
                      some_json['apparentTemperatureMax'],
                      some_json['apparentTemperatureMin'],
                      some_json['cloudCover'],
                      some_json['dewPoint'],
                      some_json['humidity'],
                      some_json['precipIntensityMax'],
                      some_json['precipAccumulation'],
                      some_json['precipTypeIsRain'],
                      some_json['precipTypeIsSnow'],
                      some_json['pressure'],
                      some_json['temperatureMax'],
                      some_json['temperatureMin'],
                      some_json['windBearing'],
                      some_json['windSpeed'],
                      some_json['NDVI']])
        a = np.reshape(a,(-1,1)).transpose()
        a = scaler_for_post_input.fit_transform(a)
        predicted = regressor.predict(a)
        yield_ = scaler_y.inverse_transform(predicted)
        print(yield_)
        return {'yield': yield_.tolist()}
    
api.add_resource(HelloWorld,'/')

if __name__ == '__main__':
    app.run(debug=True)
        