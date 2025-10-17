from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer

# Global variables
global filename
global svm_acc, random_acc, lr_acc, acc
global matrix_factor
global X_train, X_test, y_train, y_test
global epp
global classifier

svm_acc = 0
random_acc = 0
lr_acc = 0
acc = 0

courses = ['Database Developer','Portal Administrator','Systems Security Administrator','Business Systems Analyst','Software Systems Engineer',
           'Business Intelligence Analyst','CRM Technical Developer','Mobile Applications Developer','UX Designer','Quality Assurance Associate',
           'Web Developer','Information Security Analyst','CRM Business Analyst','Technical Support','Project Manager','Information Technology Manager',
           'Programmer Analyst','Design & UX','Solutions Architect','Systems Analyst','Network Security Administrator','Data Architect','Software Developer',
           'E-Commerce Analyst','Technical Services/Help Desk/Tech Support','Information Technology Auditor','Database Manager','Applications Developer',
           'Database Administrator','Network Engineer','Software Engineer','Technical Engineer','Network Security Engineer',
           'Software Quality Assurance (QA) / Testing']

def configure_button(button):
    button.config(font=('times', 12, 'bold'), bd=5, relief=RAISED, width=15)

def configure_graph_button(button):
    button.config(font=('times', 12, 'bold'), bd=5, relief=RAISED, width=15, height=2, bg="lightcoral")

def upload():
    global filename
    global matrix_factor
    filename = filedialog.askopenfilename(initialdir="dataset")
    matrix_factor = pd.read_csv(filename)
    text.insert(END, 'UCLA dataset loaded\n')
    text.insert(END, "Dataset Size: "+str(len(matrix_factor))+"\n")

def splitdataset(matrix_factor):
    le = LabelEncoder()
    normal = Normalizer()
    matrix_factor['self-learning capability?'] = pd.Series(le.fit_transform(matrix_factor['self-learning capability?']))
    matrix_factor['Extra-courses did'] = pd.Series(le.fit_transform(matrix_factor['Extra-courses did']))
    matrix_factor['certifications'] = pd.Series(le.fit_transform(matrix_factor['certifications']))
    matrix_factor['workshops'] = pd.Series(le.fit_transform(matrix_factor['workshops']))
    matrix_factor['talenttests taken?'] = pd.Series(le.fit_transform(matrix_factor['talenttests taken?']))
    matrix_factor['reading and writing skills'] = pd.Series(le.fit_transform(matrix_factor['reading and writing skills']))
    matrix_factor['memory capability score'] = pd.Series(le.fit_transform(matrix_factor['memory capability score']))
    matrix_factor['Interested subjects'] = pd.Series(le.fit_transform(matrix_factor['Interested subjects']))
    matrix_factor['interested career area'] = pd.Series(le.fit_transform(matrix_factor['interested career area']))
    matrix_factor['Job/Higher Studies?'] = pd.Series(le.fit_transform(matrix_factor['Job/Higher Studies?']))
    
    X = matrix_factor.values[:, 0:21] 
    Y = matrix_factor.values[:, 21]
    X = normal.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X, Y, X_train, X_test, y_train, y_test

def matrix():
    global X, Y, X_train, X_test, y_train, y_test
    X, Y, X_train, X_test, y_train, y_test = splitdataset(matrix_factor)
    text.insert(END, "Matrix Factorization model generated\n\n")
    text.insert(END, "Splitted Training Size for Machine Learning: "+str(len(X_train))+"\n")
    text.insert(END, "Splitted Test Size for Machine Learning: "+str(len(X_test))+"\n\n")

def SVM():
    global svm_mae
    global X, Y, X_train, X_test, y_train, y_test
    cls = svm.SVC(kernel='rbf') 
    cls.fit(X_train, y_train) 
    y_pred = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)*10
    svm_mae = mean_squared_error(y_test, y_pred) 
    text.insert(END, "SVM Accuracy: " + str(svm_acc*100) + "\n\n")

def logisticRegression():
    global classifier
    global logistic_mae
    global X, Y, X_train, X_test, y_train, y_test
    cls = LogisticRegression(penalty='l2', tol=0.02, C=3.0)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred)*10
    text.insert(END, "Logistic Regression Algorithm Accuracy: " + str(lr_acc* 100) + "\n\n")
    classifier = cls

def random():
    global random_mae
    global X, Y, X_train, X_test, y_train, y_test
    sc = StandardScaler()
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    random_acc = accuracy_score(y_test, y_pred)*10
    text.insert(END, "Random Forest Algorithm Accuracy: " + str(random_acc* 100) + "\n\n")

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

def EPP():
    global epp_mae
    global epp
    global X, Y, X_train, X_test, y_train, y_test
    base = RandomForestClassifier(n_estimators=1000, random_state=42) 
    epp = BaggingClassifier()  # Corrected here
    epp.fit(X_train, y_train)
    y_pred = epp.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 10
    text.insert(END, "EPP algorithm Accuracy: " + str(acc * 100) + "\n\n")


def predictPerformance():
    text.delete('1.0', END)

    text.insert(END, "\nPrediction Results\n")
    le = LabelEncoder()
    normal = Normalizer()
    filename = "dataset/extension_new_test_records.txt"
    test = pd.read_csv(filename)
    test['self-learning capability?'] = pd.Series(le.fit_transform(test['self-learning capability?']))
    test['Extra-courses did'] = pd.Series(le.fit_transform(test['Extra-courses did']))
    test['certifications'] = pd.Series(le.fit_transform(test['certifications']))
    test['workshops'] = pd.Series(le.fit_transform(test['workshops']))
    test['talenttests taken?'] = pd.Series(le.fit_transform(test['talenttests taken?']))
    test['reading and writing skills'] = pd.Series(le.fit_transform(test['reading and writing skills']))
    test['memory capability score'] = pd.Series(le.fit_transform(test['memory capability score']))
    test['Interested subjects'] = pd.Series(le.fit_transform(test['Interested subjects']))
    test['interested career area'] = pd.Series(le.fit_transform(test['interested career area']))
    test['Job/Higher Studies?'] = pd.Series(le.fit_transform(test['Job/Higher Studies?']))
    records = test.values[:,0:21]
    records = normal.fit_transform(records)
    value = epp.predict(records)
    text.insert(END, str(value)+"\n")
    for i in range(len(test)):
        result = value[i]
        if result <= 30:
            text.insert(END, "Predicted New Course GPA Score will be: High & Suggested/Recommended Future  is: "+courses[result]+"\n")
        if result > 30:
            text.insert(END, "Predicted New Course GPA Score will be: Low & Suggested/Recommended Future is: "+courses[result]+"\n")    


# Configure the main window
main = tk.Tk()
main.title("AI-Driven Smart Job Recommendation and Matching System") 
main.geometry("1600x1500")
main.configure(bg='orange')  # Set background color

font = ('times', 20, 'bold')
title = Label(main, text='AI-Driven Smart Job Recommendation and Matching System', font=("times", 20, 'bold'))
title.config(bg='dark blue', fg='white')
title.place(relx=0.5, y=50, anchor="center") 

# Text widget
font1 = ('times', 13, 'bold')
text = Text(main, height=15, width=130)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=170)
text.config(font=font1)

x_pos = 50

uploadButton = Button(main, text="Upload Dataset", command=upload, bg="sky blue")
configure_button(uploadButton)
uploadButton.place(x=10, y=500)

x_pos += 230  # Add space
splitButton = Button(main, text="Split Dataset", command=matrix, bg="light green")
configure_button(splitButton)
splitButton.place(x=180, y=500)

x_pos += 230  # Add space
svmButton = Button(main, text="SVM", command=SVM, bg="lightcoral")
configure_button(svmButton)
svmButton.place(x=350, y=500)

x_pos += 230  # Add space
randomButton = Button(main, text="Random Forest", command=random, bg="mediumorchid")
configure_button(randomButton)
randomButton.place(x=520, y=500)

x_pos += 230  # Add space
logisticButton = Button(main, text="Logistic Regression", command=logisticRegression, bg="lightgoldenrodyellow")
configure_button(logisticButton)
logisticButton.place(x=700, y=500)

x_pos += 230  # Add space
eppButton = Button(main, text="EPP", command=EPP, bg="palegreen")
configure_button(eppButton)
eppButton.place(x=880, y=500)

x_pos += 230  # Add space
predictButton = Button(main, text="Predict Performance", command=predictPerformance, bg="sky blue")
configure_button(predictButton)
predictButton.place(x=1100, y=500)




main.mainloop()
