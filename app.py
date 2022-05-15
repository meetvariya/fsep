from flask import Flask, render_template
from flask_navigation import Navigation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


app = Flask(__name__, template_folder="templates", static_folder='static')
nav = Navigation(app)


@app.route("/", methods=['GET','POST'])
def home():
   return render_template('index.html')

@app.route('/navpage')
def navpage():
    return render_template('navpage.html')

@app.route("/parking", methods=['GET','POST'])
def parking():
   return render_template('parking.html')

@app.route("/road_weather", methods=['GET','POST'])
def road_weather():
   return render_template('road_weather.html')

@app.route("/lane", methods=['GET','POST'])
def lane():
   return render_template('lane.html')

@app.route("/reason", methods=['GET','POST'])
def reason():
   return render_template('reason.html')

@app.route("/vehicle", methods=['GET','POST'])
def vehicle():
   return render_template('vehicle.html')

@app.route("/u_analysis", methods=['GET','POST'])
def u_analysis():
   return render_template('u_analysis.html')



@app.route("/ugraph", methods=['POST', 'GET'])
def ugraph():

    monthdata = pd.read_csv('road-accidents-in-india/only_road_accidents_data_month2.csv')
    year = request.form['year']
    state = request.form['state']
    year=int(year)
    print(year,state)
    # state = 'Delhi (Ut)'
    # year = 2013
    l = []
    months=['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY',
       'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER',
       'DECEMBER']
    tmp = monthdata[monthdata['STATE/UT'] == state]
    tmp = tmp[tmp['YEAR'] == year]
    for i in months:
        l.append(tmp[i])
    fig = plt.figure(figsize=(20,5))
    print(months)
    print(np.array(l).squeeze())
    plt.bar(months, np.array(l).squeeze())
    plt.title('Number of accidents in year ' + str(year) + ' in the state ' + state)

    plt.savefig('static/graphs/bar.png')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.pie(np.array(l).squeeze(), labels=months, autopct='%1.2f%%')
    plt.savefig('static/graphs/pie.png')


    return render_template('u_analysis.html')

@app.route('/w_visual', methods=['GET','POST'])
def w_visual():
    return render_template('w_visual.html')

@app.route('/c_visual', methods=['GET','POST'])
def c_visual():
    return render_template('c_visual.html')

@app.route('/v_visual', methods=['GET','POST'])
def v_visual():
    return render_template('v_visual.html')

@app.route("/prediction", methods=['GET','POST'])
def prediction():
   return render_template('prediction.html')

@app.route("/mlr_month", methods=['GET','POST'])
def mlr_month():
   return render_template('mlr_month.html')

@app.route("/m_predict", methods=['GET','POST'])
def m_predict():
    year = request.form['year']
    state = request.form['state']
    month = request.form['month']
    year = int(year)

    transfer_state = state
    transfer_year = year
    transfer_month = month
    #
    # data_month = pd.read_csv('road-accidents-in-india/only_road_accidents_data_month2.csv')
    #
    # # accidents per month
    # count = 0
    # month_acc =[]
    # for month in data_month.columns[2:-1]:
    #     for i in data_month.index:
    #         count = count + data_month.loc[i,month]
    #     acc = (month,count)
    #     month_acc.append(acc)
    #     count = 0
    # # month_acc
    #
    # i = 0
    #
    # new_data_month = pd.DataFrame(columns=['STATE/UT', 'YEAR', 'MONTH', 'ACCIDENTS'], index=np.arange(10000))
    # for states in data_month['STATE/UT'].unique():
    #     for year in data_month['YEAR'].unique():
    #         for month in data_month.columns[2:-1]:
    #             new_data_month.loc[i, 'STATE/UT'] = states
    #             new_data_month.loc[i, 'YEAR'] = year
    #             new_data_month.loc[i, 'MONTH'] = month
    #             i = i + 1
    #
    # j = 0
    # for i in data_month.index:
    #     for month in data_month.columns[2:-1]:
    #         new_data_month.loc[j, 'ACCIDENTS'] = data_month.loc[i, month]
    #         j = j + 1
    # new_data_month = new_data_month.dropna(how='any', axis=0)
    # new_data_month.to_csv('prepared data month.csv')
    #
    # le_month_1 = LabelEncoder()
    # le_month_2 = LabelEncoder()
    # new_data_month['STATE/UT'] = le_month_1.fit_transform(new_data_month['STATE/UT'])
    # new_data_month['MONTH'] = le_month_2.fit_transform(new_data_month['MONTH'])
    #
    # ohe = OneHotEncoder()
    # data_month_matrix_x = new_data_month[['STATE/UT', 'YEAR', 'MONTH']].values
    # data_month_matrix_y = new_data_month.ACCIDENTS
    # # data_month_matrix_x
    #
    # ohe.fit(data_month_matrix_x)
    # data_matrix_month = ohe.transform(data_month_matrix_x).toarray()
    # model = LinearRegression(fit_intercept=False)
    # X_train, X_test, y_train, y_test = train_test_split(data_matrix_month, data_month_matrix_y, test_size=0.2, random_state = 2)
    # model.fit(X_train, y_train)
    # model.score(X_test, y_test)
    #
    #
    # state = le_month_1.transform([state])
    # month = le_month_2.transform([month])
    #
    # cal = ohe.transform([[state, year, month]])
    #
    # predicted = model.predict(cal)
    data_month = pd.read_csv('road-accidents-in-india/only_road_accidents_data_month2.csv')

    count = 0
    month_acc = []
    for month in data_month.columns[2:-1]:
        for i in data_month.index:
            count = count + data_month.loc[i, month]
        acc = (month, count)
        month_acc.append(acc)
        count = 0
    month_acc

    i = 0

    new_data_month = pd.DataFrame(columns=['STATE/UT', 'YEAR', 'MONTH', 'ACCIDENTS'], index=np.arange(10000))
    for states in data_month['STATE/UT'].unique():
        for year in data_month['YEAR'].unique():
            for month in data_month.columns[2:-1]:
                new_data_month.loc[i, 'STATE/UT'] = states
                new_data_month.loc[i, 'YEAR'] = year
                new_data_month.loc[i, 'MONTH'] = month
                i = i + 1
    new_data_month.head()

    j = 0
    for i in data_month.index:
        for month in data_month.columns[2:-1]:
            new_data_month.loc[j, 'ACCIDENTS'] = data_month.loc[i, month]
            j = j + 1
    new_data_month = new_data_month.dropna(how='any', axis=0)
    new_data_month.to_csv('prepared data month.csv')
    new_data_month.head()

    le_month_1 = LabelEncoder()
    le_month_2 = LabelEncoder()
    new_data_month['STATE/UT'] = le_month_1.fit_transform(new_data_month['STATE/UT'])
    new_data_month['MONTH'] = le_month_2.fit_transform(new_data_month['MONTH'])

    ohe = OneHotEncoder(categorical_features=[0, 2])
    data_month_matrix_x = new_data_month[['STATE/UT', 'YEAR', 'MONTH']].values
    data_month_matrix_y = new_data_month.ACCIDENTS
    data_month_matrix_x

    ohe.fit(data_month_matrix_x)
    data_matrix_month = ohe.transform(data_month_matrix_x).toarray()
    model = LinearRegression(fit_intercept=False)
    X_train, X_test, y_train, y_test = train_test_split(data_matrix_month, data_month_matrix_y, test_size=0.2, random_state = 2)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    state = le_month_1.transform([state])
    month = le_month_2.transform([month])

    cal = ohe.transform([[state, year, month]])
    predicted=model.predict(cal)

    return render_template('mlr_month.html', predicted=predicted, transfer_month=transfer_month,transfer_state=transfer_state, transfer_year=transfer_year)

@app.route("/mlr_time", methods=['GET','POST'])
def mlr_time():
   return render_template('mlr_time.html')
@app.route("/t_predict", methods=['GET','POST'])
def t_predict():
    year = request.form['year']
    state = request.form['state']
    time = request.form['time']
    year = int(year)

    transfer_state = state
    transfer_time = time
    data = pd.read_csv('time_prepared_data.csv')
    data = data.drop('Unnamed: 0', axis=1)
    le1 = LabelEncoder()
    le2 = LabelEncoder()

    for i in data.index:
        if data.loc[i, 'TIME'] == '0-3 hrs. (Night)':
            data.loc[i, 'TIME'] = 0
        elif data.loc[i, 'TIME'] == '3-6 hrs. (Night)':
            data.loc[i, 'TIME'] = 1
        elif data.loc[i, 'TIME'] == '6-9 hrs (Day)':
            data.loc[i, 'TIME'] = 2
        elif data.loc[i, 'TIME'] == '9-12 hrs (Day)':
            data.loc[i, 'TIME'] = 3
        elif data.loc[i, 'TIME'] == '12-15 hrs (Day)':
            data.loc[i, 'TIME'] = 4
        elif data.loc[i, 'TIME'] == '15-18 hrs (Day)':
            data.loc[i, 'TIME'] = 5
        elif data.loc[i, 'TIME'] == '18-21 hrs (Night)':
            data.loc[i, 'TIME'] = 6
        elif data.loc[i, 'TIME'] == '21-24 hrs (Night)':
            data.loc[i, 'TIME'] = 7

    data['STATE/UT'] = le1.fit_transform(data['STATE/UT'])

    ohe = OneHotEncoder(categorical_features=[0])
    data_matrix_x = data[['STATE/UT', 'YEAR', 'TIME']].values
    data_matrix_y = data.ACCIDENTS

    ohe.fit(data_matrix_x)
    data_matrix = ohe.transform(data_matrix_x).toarray()

    model2 = LinearRegression(fit_intercept=False)
    X_train, X_test, y_train, y_test = train_test_split(data_matrix, data_matrix_y, test_size=0.2, random_state = 2)
    model2.fit(X_train, y_train)



    state1 = le1.transform([state])
    if time == '0-3 hrs. (Night)':
        time = 0
    elif time == '3-6 hrs. (Night)':
        time = 1
    elif time == '6-9 hrs (Day)':
        time = 2
    elif time == '9-12 hrs (Day)':
        time = 3
    elif time == '12-15 hrs (Day)':
        time = 4
    elif time == '15-18 hrs (Day)':
        time = 5
    elif time == '18-21 hrs (Night)':
        time = 6
    elif time == '21-24 hrs (Night)':
        time = 7
    cal = ohe.transform([[state1, year, time]])
    final_prediction = model2.predict(cal)

    return render_template('mlr_time.html', final_prediction=final_prediction,transfer_state=transfer_state,transfer_time=transfer_time,year=year)

@app.route("/slr", methods=['GET','POST'])
def slr():
   return render_template('slr.html')

@app.route("/s_predict", methods=['GET','POST'])
def s_predict():
    year1 = request.form['year']
    transfer_year=year1
    data = pd.read_csv('road-accidents-in-india/only_road_accidents_data3.csv')

    e = []
    cnt2 = 0
    for year in data['YEAR'].unique():
        for i in data.index:
            if data.loc[i, 'YEAR'] == year:
                cnt2 = cnt2 + data.loc[i, 'Total']
        year_acc = (year, cnt2)
        cnt2 = 0
        e.append(year_acc)
    model = LinearRegression()
    X_data = np.array([t[0] for t in e])
    Y_data = np.array([y[1] for y in e])

    model.fit(X_data.reshape(len(X_data), 1), Y_data.reshape(len(Y_data), 1))
    #model.score(X_data.reshape(len(X_data), 1), Y_data.reshape(len(Y_data), 1))
    final_prediction = model.predict([[float(year1)]])
    print(final_prediction)

    return render_template('slr.html', final_prediction=final_prediction, transfer_year=transfer_year)



if __name__ == "__main__":
  app.run(debug=True)