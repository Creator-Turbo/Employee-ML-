from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\Employee\\Employee-ML-\\employee_classification\\Employee.csv')
X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

# Basic encoding and model setup
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = {
        'Education': data['education'],
        'Joining Year': int(data['joining_year']),
        'City': data['city'],
        'Payment Tier': int(data['payment_tier']),
        'Age': int(data['age']),
        'Gender': data['gender'],
        'Ever Benched': data['ever_benched'],
        'Experience in Current Domain': int(data['experience'])
    }
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
