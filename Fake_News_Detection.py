from operator import add
from typing import Any, Collection
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

#database
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db= firestore.client()


#tfid


app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x.values.astype('U'), y.values.astype('U'), test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

def accuracy(news):
  tfid_x_train = tfvect.fit_transform(x_train)
  tfid_x_test = tfvect.transform(x_test)
  classifier = PassiveAggressiveClassifier(max_iter=50)
  classifier.fit(tfid_x_train,y_train)
  y_pred = classifier.predict(tfid_x_test)
  score = accuracy_score(y_test,y_pred)
  return score

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        message = request.form['message1']
        message2 = request.form['message2']
        pred1 = fake_news_det(message)
        pred2 = fake_news_det(message2)
        count_accuracy= accuracy(message)
        accuracy_score= round(count_accuracy*100,2)


        if pred1 == ['FAKE'] and pred2 == ['FAKE']:
          db.collection('Summary').add({'label':'Fake'})

        elif pred1 == ['REAL'] and pred2 == ['FAKE']:
          db.collection('Summary').add({'label':'Fake'})

        elif pred1 == ['FAKE'] and pred2 == ['REAL']:
          db.collection('Summary').add({'label':'Fake'})

        elif pred1 == ['REAL'] and pred2 == ['REAL']:
          db.collection('Summary').add({'label':'Real'})

        else:
          print("error")
        
        #count collections
        
        total=0
        summary = db.collection("Summary").list_documents()
        for count in summary:
            total += 1
        print("Number of test conducted",total)
    
        docs = db.collection("Summary").where("label", "==", "Fake").stream()
        totalfake=0
        for count in docs:
            totalfake += 1
        print("Fake news:",totalfake)

        docs = db.collection("Summary").where("label", "==", "Real").stream()
        totalReal=0
        for count in docs:
            totalReal += 1
        print("Real news:",totalReal)

        print(pred1)
        print(pred2)
        print(accuracy_score)
        
        return render_template('Home.html', prediction1=pred1,prediction2=pred2,total=total,Real=totalReal,Fake=totalfake,score=accuracy_score)
        
    else:
        return render_template('Home.html', prediction1="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)

