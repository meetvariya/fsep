import tkinter as tk
from tkinter import ttk
from PIL import Image ,ImageTk

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

r=tk.Tk()
r.geometry("1920x1080+30+30")
r.title('Road Accident Analysis using Machine Learning')

background_image=ImageTk.PhotoImage(Image.open('background.jpg'))
background_label = tk.Label(r, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0, relwidth=1, relheight=1)

tt1=tk.Label(r,text='    Road Accidents Analysis and Prediction',width=50)
tt1.config(font=("Ariel",50))
tt1.pack(padx=10,pady=100)


new_data=pd.read_csv('time_prepared_data.csv')
data = pd.read_csv('old_time_data.csv')
e= []
cnt2 = 0
for year in data['YEAR'].unique():
    for i in data.index:
        if data.loc[i,'YEAR'] == year:
            cnt2 = cnt2 + data.loc[i,'Total'] 
    year_acc = (year,cnt2)
    cnt2=0
    e.append(year_acc)
         
model = LinearRegression()
X_data = np.array([t[0] for t in e])
Y_data = np.array([y[1] for y in e])
model.fit(X_data.reshape(len(X_data),1),Y_data.reshape(len(Y_data),1))

data = pd.read_csv('time_prepared_data.csv')
data =data.drop('Unnamed: 0',axis=1)
le1 = LabelEncoder()
le2 = LabelEncoder()
        
for i in data.index:
    if data.loc[i,'TIME']== '0-3 hrs. (Night)':
         data.loc[i,'TIME'] = 0
    elif data.loc[i,'TIME']== '3-6 hrs. (Night)':
         data.loc[i,'TIME'] = 1
    elif data.loc[i,'TIME']== '6-9 hrs (Day)':
         data.loc[i,'TIME'] = 2
    elif data.loc[i,'TIME']== '9-12 hrs (Day)':
         data.loc[i,'TIME'] = 3
    elif data.loc[i,'TIME']== '12-15 hrs (Day)':
         data.loc[i,'TIME'] = 4
    elif data.loc[i,'TIME']== '15-18 hrs (Day)':
         data.loc[i,'TIME'] = 5
    elif data.loc[i,'TIME']== '18-21 hrs (Night)':
         data.loc[i,'TIME'] = 6
    elif data.loc[i,'TIME']== '21-24 hrs (Night)':
         data.loc[i,'TIME'] = 7
        
data['STATE/UT']=le1.fit_transform(data['STATE/UT'])

ohe = OneHotEncoder(categorical_features=[0])
data_matrix_x= data[['STATE/UT','YEAR','TIME']].values
data_matrix_y = data.ACCIDENTS

ohe.fit(data_matrix_x)
data_matrix = ohe.transform(data_matrix_x).toarray()
    
model2 = LinearRegression(fit_intercept=False)
X_train ,X_test ,y_train, y_test = train_test_split(data_matrix,data_matrix_y ,test_size = 0.2)
model2.fit(X_train,y_train)

new_data_month = pd.read_csv('month_prepared_data.csv')
le_month_1 = LabelEncoder()
le_month_2 = LabelEncoder()
new_data_month['STATE/UT']=le_month_1.fit_transform(new_data_month['STATE/UT'])
new_data_month['MONTH']=le_month_2.fit_transform(new_data_month['MONTH'])


ohe_month = OneHotEncoder(categorical_features=[0,2])
data_month_matrix_x= new_data_month[['STATE/UT','YEAR','MONTH']].values
data_month_matrix_y = new_data_month.ACCIDENTS


ohe_month.fit(data_month_matrix_x)
data_matrix_month = ohe_month.transform(data_month_matrix_x).toarray()
model3 = LinearRegression(fit_intercept=False)
X_train_month ,X_test_month ,y_train_month, y_test_month = train_test_split(data_matrix_month,data_month_matrix_y ,test_size = 0.2)
model3.fit(X_train_month,y_train_month)

def predict():
    year=0
    time=0
    state=0

    top2=tk.Toplevel(r)
    top2.geometry("1920x1080+30+30")
    top2.title('Prediction')

    tt1=tk.Label(top2,text='   Prediction',width=80)
    tt1.config(font=("Ariel",50))
    tt1.pack(padx=10,pady=60)

    def pb_prints(msg,a,b):

        txt=tk.Toplevel(top2)
        txt.geometry("640x400+30+30")

        tt1=tk.Message(txt,text=msg,relief=tk.RAISED,anchor=tk.W,width=1000)
        tt1.config(font=("Ariel",20))
        tt1.pack(padx=10,pady=5)
        tt2=tk.Message(txt,text=a,relief=tk.RAISED,anchor=tk.W,width=800)
        tt2.config(font=("Ariel",20))
        tt2.pack(padx=10,pady=5)
        c="Accuracy: "+str(b*100)+"%"
        tt3=tk.Message(txt,text=c,relief=tk.RAISED,anchor=tk.W,width=800)
        tt3.config(font=("Ariel",20))
        tt3.pack(padx=10,pady=5)
        b4=tk.Button(txt,text='Back',width=10,height=2,command=txt.destroy)
        b4.config(font=("Ariel",20))
        b4.pack(side='bottom',pady=2)

    def entry(z):
        year = int(z.get())
        acc_year=model.predict([[year]])
        pb_prints('Predicted number of accidents',acc_year[0],model.score(X_data.reshape(len(X_data),1),Y_data.reshape(len(Y_data),1)))
        
    def ShowReg(top2):
        txt=tk.Toplevel(top2)
        txt.geometry("1280x720+30+30")
        img = ImageTk.PhotoImage(Image.open('model1_line.png'))
        panel = tk.Label(txt, image = img)
        panel.image = img
        panel.place(x=0,y=0)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        txt.mainloop()
        
    def pb1():
        txt=tk.Toplevel(top2)
        txt.geometry("1280x720+30+30")
        tt1=tk.Message(txt,text='Enter the Year to Predict number of Accidents',relief=tk.RAISED,anchor=tk.W,width=1000)
        tt1.config(font=("Ariel",20))
        tt1.pack(padx=20,pady=20)
        z = tk.Entry(txt,font =("Ariel",15))
        z.pack()
        z.focus_set()
        b4=tk.Button(txt,text='Back',width=10,height=2,command=txt.destroy)
        b4.config(font=("Ariel",20))
        b4.pack(side='bottom',pady=2)
        b = tk.Button(txt,text='Predict',command=lambda:entry(z),activeforeground='red',width=50,height=2)
        b.config(font=("Ariel",20))
        b.pack(side='bottom')
        bt = tk.Button(txt,text='Show regression Line',command=lambda:ShowReg(top2),activeforeground='red',width=50,height=2)
        bt.config(font=("Ariel",20))
        bt.pack(side='bottom')
        

    
    def entry1(comboboxTime,comboboxState,comboboxYear):
        year = int(comboboxYear.get())
        state = comboboxState.get()
        time = comboboxTime.get()
        state1=le1.transform([state])
        if time== '0-3 hrs. (Night)':
             time = 0
        elif time== '3-6 hrs. (Night)':
             time = 1
        elif time== '6-9 hrs (Day)':
             time = 2
        elif time== '9-12 hrs (Day)':
             time = 3
        elif time== '12-15 hrs (Day)':
             time = 4
        elif time== '15-18 hrs (Day)':
             time = 5
        elif time== '18-21 hrs (Night)':
             time = 6
        elif time== '21-24 hrs (Night)':
             time = 7
        cal = ohe.transform([[state1,year,time]])     
        res=model2.predict(cal)
        if res[0] < 0:
            res[0]=0
        pb_prints('Predicted number of accidents',res[0],model2.score(X_train,y_train))
        
    def entry2(comboboxMonth,comboboxState,comboboxYear):
        year = int(comboboxYear.get())
        state = comboboxState.get()
        month = comboboxMonth.get()
        
        state = le_month_1.transform([state])
        month = le_month_2.transform([month])
        cal = ohe_month.transform([[state,year,month]])
        res=model3.predict(cal)
        if res[0] < 0:
            res[0]=0
        pb_prints('Predicted number of accidents',res[0],model3.score(X_train_month,y_train_month))
        
    def pb2():
        txt=tk.Toplevel(top2)
        txt.geometry("1280x720+30+30")
        txt.title('Multipe Linear Regression')
        
        tt1=tk.Message(txt,text='Select the Year',relief=tk.RAISED,anchor=tk.W,width=300)
        tt1.config(font=("Ariel",20))
        year = tk.StringVar()
        tt1.pack(padx=20,pady=20)
        comboboxYear = ttk.Combobox(txt , width =20 ,textvariable = year,font =("Ariel",15))
        comboboxYear['values']=list(range(2000,2025))
        comboboxYear.pack(padx=20,pady=20)


        tt2=tk.Message(txt,text='Select the State',relief=tk.RAISED,anchor=tk.W,width=300)
        tt2.config(font=("Ariel",20))
        state = tk.StringVar()
        tt2.pack(padx=20,pady=20)
        comboboxState = ttk.Combobox(txt , width =20 ,textvariable = state,font =("Ariel",15))
        comboboxState['values'] = list(new_data['STATE/UT'].unique())
        comboboxState.pack(padx=20,pady=20)


        tt3=tk.Message(txt,text='Select the Timespan',relief=tk.RAISED,anchor=tk.W,width=300)
        tt3.config(font=("Ariel",20))
        tt3.pack(padx=10,pady=5)
        
        time = tk.StringVar()
        comboboxTime = ttk.Combobox(txt , width =30 ,textvariable = time,font =("Ariel",15))
        comboboxTime['values'] = list(new_data['TIME'].unique())
        comboboxTime.pack(padx=15,pady=5)
        b4=tk.Button(txt,text='Back',width=10,height=2,command=txt.destroy)
        b4.config(font=("Ariel",20))
        b4.pack(side='bottom',pady=2)
        b = tk.Button(txt,text='Predict',command=lambda:entry1(comboboxTime,comboboxState,comboboxYear),activeforeground='red',width=50,height=2)
        b.config(font=("Ariel",20))
        b.pack(side='bottom')

    def pb3():
        new_data_month = pd.read_csv('month_prepared_data.csv')
        txt=tk.Toplevel(top2)
        txt.geometry("1280x720+30+30")
        txt.title('Multipe Linear Regression')
        
        tt1=tk.Message(txt,text='Select the Year',relief=tk.RAISED,anchor=tk.W,width=300)
        tt1.config(font=("Ariel",20))
        tt1.pack(padx=20,pady=20)
        
        year = tk.StringVar()
        comboboxYear = ttk.Combobox(txt , width =20 ,textvariable = year,font =("Ariel",15))
        comboboxYear['values']=list(range(2000,2025))
        comboboxYear.pack(padx=20,pady=20)


        tt2=tk.Message(txt,text='Select the State',relief=tk.RAISED,anchor=tk.W,width=300)
        tt2.config(font=("Ariel",20))
        tt2.pack(padx=20,pady=20)
        
        state = tk.StringVar()
        comboboxState = ttk.Combobox(txt , width =20 ,textvariable = state,font =("Ariel",15))
        comboboxState['values'] = list(new_data_month['STATE/UT'].unique())
        comboboxState.pack(padx=20,pady=20)


        tt3=tk.Message(txt,text='Select the Month',relief=tk.RAISED,anchor=tk.W,width=300)
        tt3.config(font=("Ariel",20))
        tt3.pack(padx=20,pady=20)
        
        month = tk.StringVar()
        comboboxMonth = ttk.Combobox(txt , width =20 ,textvariable = month,font =("Ariel",15))
        comboboxMonth['values'] = list(new_data_month['MONTH'].unique())
        comboboxMonth.pack(padx=20,pady=20)
        b4=tk.Button(txt,text='Back',width=10,height=2,command=txt.destroy)
        b4.config(font=("Ariel",20))
        b4.pack(side='bottom',pady=2)
        b = tk.Button(txt,text='Predict',command=lambda:entry2(comboboxMonth,comboboxState,comboboxYear),activeforeground='red',width=50,height=2)
        b.config(font=("Ariel",20))
        b.pack(side='bottom')
        
    pb1=tk.Button(top2,text='Simple Linear Regression',width=50,height=1,command=pb1)
    pb1.config(font=("Ariel",20))
    pb1.pack(pady=10,padx=10)
    
    pb2=tk.Button(top2,text='Multiple Linear Regression(time)',width=50,height=1,command=pb2)
    pb2.config(font=("Ariel",20))
    pb2.pack(pady=15,padx=10)

    pb3=tk.Button(top2,text='Multiple Linear Regression(month)',width=50,height=1,command=pb3)
    pb3.config(font=("Ariel",20))
    pb3.pack(pady=20,padx=10)
        
    pb0=tk.Button(top2,text='Back',width=10,height=2,command=top2.destroy)
    pb0.pack(pady=30,padx=10)

    
    top2.mainloop()    
    

button2= tk.Button(r,text='Prediction',width=25,height=2,command=predict)
button2.config(font=("Ariel",20))
button2.pack(pady=30)

button3= tk.Button(r,text='Exit',width=25,height=2,command=r.destroy)
button3.config(font=("Ariel",20))
button3.pack(pady=30)

r.mainloop()
