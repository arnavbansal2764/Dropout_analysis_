import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

data = pd.read_csv('data.csv')

label_encoders = {}
categorical_columns = ['school', 'area', 'gender', 'caste']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('dropout', axis=1)
y = data['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
def predict(school,area,gender,caste,age):
    school_encoded = label_encoders['school'].transform([school])[0]
    area_encoded = label_encoders['area'].transform([area])[0]
    gender_encoded = label_encoders['gender'].transform([gender])[0]
    caste_encoded = label_encoders['caste'].transform([caste])[0]

    input_data = [[school_encoded, area_encoded, gender_encoded, caste_encoded, age]]
    dropout_prob = clf.predict_proba(input_data)[:, 1]
    return dropout_prob
    # print(f"{dropout_prob[0]:.2f}")


prob=0
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global prob
    if request.method == 'POST':
        # global school, area, gender, caste, age
        school = request.form.get('school')
        area = request.form.get('area')
        gender = request.form.get('gender')
        caste = request.form.get('caste')
        age = request.form.get('age')
        # print(school, area, gender, caste, age)
        prob = predict(school,area,gender,caste,age)
        # print(prob)
   
    return render_template('index_old.html',variable = prob)
@app.route('/aboutus.html')
def about():
    return render_template('/aboutus.html')

@app.route('/index.html')
def index():
    return render_template('/index.html')

@app.route('/trends.html')
def trends():       
    return render_template('/trends.html')


@app.route('/index_old.html')
def check():
    return render_template('/index_old.html')
@app.route('/Templates/aboutus.html')
def aboutUS():
    return render_template('/aboutus.html')

@app.route('/Templates/index.html')
def tempindex():
    return render_template('/index.html')

@app.route('/Templates/trends.html')
def temptrend():
    
    return render_template('trends.html')

@app.route('/Templates/index_old.html')
def tempcheck():
    return render_template('/index_old.html')
@app.route('/Templates/style.css')
def tempstyle():
    return render_template('/style.css')
@app.route('/style.css')
def style():
    return render_template('/style.css')


if __name__ == "__main__":
    app.run(debug=True, port=8000)