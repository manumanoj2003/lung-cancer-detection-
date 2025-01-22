import os
import pickle
from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
MODEL_PATH = './model/random_forest.pkl'
SCALER_PATH = './model/scaler.pkl'
LABEL_ENCODER_CANCER_PATH = './model/label_encoder_cancer.pkl'

# Train and save the model (only for the first run)
if not os.path.exists(MODEL_PATH):
    os.makedirs('./model', exist_ok=True)
    data = pd.read_csv('./survey lung cancer.csv')

    # Encode target column (Yes/No -> 1/0)
    label_encoder_cancer = LabelEncoder()
    data['LUNG_CANCER'] = label_encoder_cancer.fit_transform(data['LUNG_CANCER'])

    # Encode gender column (Male/Female -> 1/2)
    data['GENDER'] = data['GENDER'].map({'Male': 1, 'Female': 2})

    # Separate features and target
    X = data.drop(columns=['LUNG_CANCER'])
    y = data['LUNG_CANCER']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    # Save the model and preprocessors
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(LABEL_ENCODER_CANCER_PATH, 'wb') as f:
        pickle.dump(label_encoder_cancer, f)

# Load model and preprocessors
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_ENCODER_CANCER_PATH, 'rb') as f:
    label_encoder_cancer = pickle.load(f)

# Input form columns
FORM_COLUMNS = list(pd.read_csv('./survey lung cancer.csv').columns[:-1])  # Exclude LUNG_CANCER

# Default login credentials
USERNAME = "admin"
PASSWORD = "admin"

@app.route('/')
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    if username == USERNAME and password == PASSWORD:
        session['username'] = username
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error="Invalid credentials. Please try again.")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', columns=FORM_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        # Collect user input
        user_input = {column: request.form[column] for column in FORM_COLUMNS}

        # Map Yes/No dropdown to numeric values
        for key in user_input.keys():
            if key == 'GENDER':
                user_input[key] = 1 if user_input[key] == 'Male' else 2
            elif user_input[key] == 'Yes':
                user_input[key] = 2
            elif user_input[key] == 'No':
                user_input[key] = 1

        # Create DataFrame
        input_data = pd.DataFrame([user_input])

        # Scale the input data
        X_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(X_input)
        result = label_encoder_cancer.inverse_transform(prediction)[0]  # Convert numeric result back to "Yes"/"No"

        return render_template('result.html', result=result, data=request.form)

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
