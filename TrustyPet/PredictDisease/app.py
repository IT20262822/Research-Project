from flask_socketio import SocketIO
from flask import Flask, render_template, request, redirect, url_for, session, make_response
from flask_cors import cross_origin, CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from numpy import array
from json import dumps
from os import environ
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
from sklearn.preprocessing import MultiLabelBinarizer


app = Flask(__name__)
app.secret_key = '09cd6cb8206a12b54a7ddb28566be757'
socketio = SocketIO(app, cors_allowed_origins="*")
bcrypt = Bcrypt(app)

cluster = MongoClient(
        "mongodb://localhost:27017/")
db = cluster['TrustyPet']
db = cluster['TrustyPet']

fields = []
description = {}

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/bot')
@cross_origin()
def chat_bot():
    if "email" in session:
        return render_template('bot.html')
    else:
        return redirect(url_for('login', next='/bot'))        

@app.route('/signup', methods=['POST'])
@cross_origin()
def sign_up():
    collection = db['users']
    form_data = request.form
    pw_hash =  bcrypt.generate_password_hash(form_data['password']).decode('utf-8')
    result = collection.find_one({'email': form_data['email']})
    if result == None:
        id = collection.insert_one({
            'name': form_data['name'],
            'email': form_data['email'],
            'password': pw_hash
        }).inserted_id
        response = ''
        if id == '':
            response = 'failed'
        else:
            response = 'success'
    else:
        response = 'failed'
    return render_template('signup.html', response=response)


@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        collection = db['users']
        form_data = request.form
        next_url = request.form.get('next')
        query_data = {
            "email": form_data['email']
        }
        result = collection.find_one(query_data)
        response = ''
        if result == None:
            response = 'failed'
        else:
            pw = form_data['password']
            if bcrypt.check_password_hash(result['password'], pw) == True:
                response = 'succeeded'
                session['email'] = form_data['email']
                session['name'] = result['name']
                
            else:
                response = 'failed'
        
        if response == 'failed':
            return render_template('login.html', response=response)
        else:
            if next_url:
                return redirect(next_url)
            else:
                return redirect(url_for('chat_bot'))
    else:
        if "email" in session:
            return redirect(url_for("chat_bot"))
        else:
            return render_template('login.html')
        
@app.route('/dashboard')
@cross_origin()
def dashboard():
    if "email" in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
@cross_origin()
def logout():
    if 'email' in session:
        session.pop("email", None)
        session.pop("username", None)
        
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))

@app.route('/bot',methods=['POST'])
@cross_origin()
def bot():
   
    # Load the dataset
    data = pd.read_csv('medical_data.csv')
    symptom_features = list(set(data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].values.ravel()))
    # Convert the Symptoms column to a list of lists for one-hot encoding
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].apply(lambda x: x.str.split(','))
   
   # Flatten the symptom lists
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].apply(lambda x: [item for sublist in x for item in sublist])    
    symptoms = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].values.tolist()   
    
    # Initialize a MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(classes=symptom_features)
    mlb.fit(symptoms)
    symptoms_encoded = mlb.transform(symptoms)

    # One-hot encoded symptoms
    X = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)
    y = data['Disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    #Handling User Input
    user_data = request.json  
    symptoms_input = user_data.get('symptoms', "").split(',')
    print(user_data)
    
    
    # Encoding the user's symptoms using the same MultiLabelBinarizer
    user_symptoms_encoded = mlb.transform([symptoms_input])

    # One-hot encoded symptoms for prediction
    user_input = pd.DataFrame([list(user_symptoms_encoded[0])])

    # Predicting the disease using the trained model
    predicted_disease = classifier.predict(user_input)
    print(predicted_disease)
    return jsonify({'predicted_disease': predicted_disease[0]})

@app.route('/getinfo',methods=['POST'])
@cross_origin()
def getinfo():
    # Accessing URL-encoded parameters from the query string
    param1 = request.args.get('disease')
    
    description_data = pd.read_csv('description.csv')
    treatment_data = pd.read_csv('treatment.csv')
    
    # Fetch treatment and description for predicted disease
    predicted_disease = predicted_disease[0]
    predicted_treatment = treatment_data.loc[treatment_data['Disease'] == predicted_disease]['Treatment'].values[0]
    predicted_description = description_data.loc[description_data['Disease'] == predicted_disease]['Description'].values[0]

    # Construct response
    response = {
        'predicted_disease': predicted_disease,
        'treatment': predicted_treatment,
        'description': predicted_description
    }

    return jsonify(response)


@app.route('/info')
def example():
    # Accessing URL-encoded parameters from the query string
    disease = request.args.get('disease')
    
    return render_template("info.html", disease=disease)







#checks if the Python script is being run directly
if __name__ == '__main__':
    app.run(debug=True)



