from flask import Flask, render_template, request
import sqlite3
import numpy as np
from joblib import load
from feature import FeatureExtraction
import warnings
import os

warnings.filterwarnings('ignore')

# -------------------------------
# Load the trained model using joblib
# -------------------------------
MODEL_PATH = os.path.join("model", "model.pkl")


from joblib import load
gbc = load(MODEL_PATH)


# -------------------------------
# Initialize SQLite database
# -------------------------------
DB_PATH = 'user_data.db'
with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user(
            name TEXT, 
            password TEXT, 
            mobile TEXT, 
            email TEXT
        )
    """)
    connection.commit()

# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/userlog', methods=['POST'])
def userlog():
    """User login with safe SQLite query"""
    name = request.form['name']
    password = request.form['password']

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name, password FROM user WHERE name=? AND password=?", 
            (name, password)
        )
        result = cursor.fetchone()

    if result:
        return render_template('home.html')
    else:
        return render_template('index.html', msg='Sorry, Incorrect Credentials Provided. Try Again')

@app.route('/userreg', methods=['POST'])
def userreg():
    """User registration with safe SQLite insert"""
    name = request.form['name']
    password = request.form['password']
    mobile = request.form['phone']
    email = request.form['email']

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO user VALUES (?, ?, ?, ?)", 
            (name, password, mobile, email)
        )
        connection.commit()

    return render_template('index.html', msg='Successfully Registered')

@app.route('/IDS', methods=['GET', 'POST'])
def IDS():
    """Phishing URL detection using GradientBoostingClassifier"""
    if request.method == 'POST':
        url = request.form['Link']
        print("URL received:", url)

        # Feature extraction
        obj = FeatureExtraction(url)
        features = obj.getFeaturesList()
        feature_details = obj.getFeatureDetails()

        if len(features) != 30:
            return render_template("ml.html", xx=-1, msg="Feature extraction failed. Expected 30 features.")

        x = np.array(features).reshape(1, -1)

        # Prediction
        try:
            y_pred = gbc.predict(x)[0]
            proba = gbc.predict_proba(x)[0]
            classes = gbc.classes_

            # Map probabilities safely
            y_pro_phishing = proba[list(classes).index(-1)] if -1 in classes else proba[0]
            y_pro_non_phishing = proba[list(classes).index(1)] if 1 in classes else proba[1]

            pred_msg = "It is {0:.2f}% safe to use.".format(y_pro_non_phishing * 100)
            print(f"\nPrediction: {pred_msg}\n")

            return render_template(
                'ml.html', 
                xx=round(y_pro_non_phishing, 2),
                res=round(y_pro_non_phishing, 2),
                url=url,
                pred=pred_msg,
                feature_details=feature_details
            )
        except Exception as e:
            return render_template("ml.html", xx=-1, msg=f"Prediction error: {str(e)}")

    return render_template("ml.html", xx=-1, msg="You are in ML page")

@app.route('/logout')
def logout():
    return render_template('index.html')

# -------------------------------
# Run the Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
