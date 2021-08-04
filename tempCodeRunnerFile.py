from operator import add
from typing import Any, Collection
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import Flask, render_template, request
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
dataframe = pd.read_csv('news.csv', low_memory=False)
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train.astype('U').values)
    tfid_x_test = tfvect.transform(x_test.astype('U').values)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        if pred == ['FAKE']:
          db.collection('Summary').add({'label':'Fake'})
        else:   
          db.collection('Summary').add({'label':'Real'})
        
        #count collections
        summary = db.collection("Summary").list_documents()
        total=0
        
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

        print(pred)

        return render_template('index.html', prediction=pred,total=total,Real=totalReal,Fake=totalfake)


    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)

