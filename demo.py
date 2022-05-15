from flask import Flask, render_template
import requests
from flask import Flask, render_template, url_for, jsonify, request, redirect, session, flash, Response, make_response

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates", static_folder='statics')

@app.route("/", methods=['POST', 'GET'])
def home():
   return render_template('demo.html')

# @app.route("/ugraph", methods=['POST', 'GET'])
# def ugraph():
#
#     monthdata = pd.read_csv('road-accidents-in-india/only_road_accidents_data_month2.csv')
#     year = request.form['year']
#     state = request.form['state']
#     year=int(year)
#     print(year,state)
#     # state = 'Delhi (Ut)'
#     # year = 2013
#     l = []
#     months=['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY',
#        'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER',
#        'DECEMBER']
#     tmp = monthdata[monthdata['STATE/UT'] == state]
#     tmp = tmp[tmp['YEAR'] == year]
#     for i in months:
#         l.append(tmp[i])
#     fig = plt.figure(figsize=(20,5))
#     print(months)
#     print(np.array(l).squeeze())
#     plt.bar(months, np.array(l).squeeze())
#     plt.title('Number of accidents in year ' + str(year) + ' in the state ' + state)
#
#     plt.savefig('statics/graphs/accidents.png')
#
#     return render_template('demo.html')




if __name__ == "__main__":
  app.run()