from flask import Flask, render_template, request
import joblib
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the saved model
model = joblib.load('notebook/best_model_MLP.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # # Map Gender and EverBenched to integers
    # gender_map = {'Male': 0, 'Female': 1}
    # benched_map = {'Yes': 1, 'No': 0}
    # education_map={'Bachelors':1,'Masters':2}

    features = {
    'Age': [int(data['age'])],
    'Gender': [(data['gender'])],
    'PaymentTier': [int(data['payment_tier'])],
    'ExperienceInCurrentDomain': [int(data['experience'])],
    'Education': [(data['education'])],
    'City': [(data['city'])], # for the some time using integer for this 
    'JoiningYear': [int(data['joining_year'])],
    'EverBenched': [(data['ever_benched'])]

}

# # Create a dictionary for the input features
#     features = {
#             'Age': [int(data.get('age', 0))],
#             'Gender': [gender_map.get(data.get('gender', 'Male'))],
#             'PaymentTier': [int(data.get('payment_tier', 0))],
#             'ExperienceInCurrentDomain': [int(data.get('experience', 0))],
#             'Education': [education_map.get('Bachelors','Masters')],
#             'City': [data.get('city', '')],
#             'JoiningYear': [int(data.get('joining_year', 0))],
#             'EverBenched': [benched_map.get(data.get('ever_benched', 'No'))]
#         }



    # Convert the features to a DataFrame
    features_df = pd.DataFrame(features)
    # # Ensure the correct data types for each column
    # features_df['City'] = features_df['City'].astype(str)
     
    features_df=features_df.dropna()
    # Apply one-hot encoding on the specified columns (ensure to pass a list of column names)
    features_df1 = pd.get_dummies(features_df, columns=['Gender', 'Education', 'City', 'EverBenched'])
    


    # Make prediction
    prediction = model.predict(features_df)[0]

    prediction_text = 'Leave' if prediction == 1 else 'Stay'

    # Render result.html and pass the prediction result
    return render_template('result.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

