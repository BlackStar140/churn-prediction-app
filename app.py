
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = pickle.load(open('churn_prediction_model.pkl', 'rb'))
onehot_encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('standard_scaler.pkl')

def tenure_group(tenure):
    if tenure <= 12:
        return '0-1 year'
    elif tenure <= 24:
        return '1-2 years'
    elif tenure <= 48:
        return '2-4 years'
    elif tenure <= 60:
        return '4-5 years'
    else:
        return '5+ years'

@app.route('/', methods=['GET', 'POST'])
def predict_churn():
    if request.method == 'POST':
        input_data = {
            'gender': [1 if request.form['gender'] == 'male' else 0],
            'SeniorCitizen': [1 if request.form['SeniorCitizen'] == 'Yes' else 0],
            'Partner': [1 if request.form['Partner'] == 'Yes' else 0],
            'Dependents': [1 if request.form['Dependents'] == 'Yes' else 0],
            'tenure': [int(request.form['tenure'])],
            'PhoneService': [1 if request.form['PhoneService'] == 'Yes' else 0],
            'MultipleLines': [2 if request.form['MultipleLines'] == 'Yes' else 
                              (1 if request.form['MultipleLines'] == 'No phone service' else 0)],
            'InternetService': [0 if request.form['InternetService'] == 'DSL' else 
                                (1 if request.form['InternetService'] == 'Fiber optic' else 2)],
            'OnlineSecurity': [2 if request.form['OnlineSecurity'] == 'Yes' else 
                               (1 if request.form['OnlineSecurity'] == 'No internet service' else 0)],
            'OnlineBackup': [2 if request.form['OnlineBackup'] == 'Yes' else 
                             (1 if request.form['OnlineBackup'] == 'No internet service' else 0)],
            'DeviceProtection': [2 if request.form['DeviceProtection'] == 'Yes' else 
                                 (1 if request.form['DeviceProtection'] == 'No internet service' else 0)],
            'TechSupport': [2 if request.form['TechSupport'] == 'Yes' else 
                            (1 if request.form['TechSupport'] == 'No internet service' else 0)],
            'StreamingTV': [2 if request.form['StreamingTV'] == 'Yes' else 
                            (1 if request.form['StreamingTV'] == 'No internet service' else 0)],
            'StreamingMovies': [2 if request.form['StreamingMovies'] == 'Yes' else 
                                (1 if request.form['StreamingMovies'] == 'No internet service' else 0)],
            'Contract': [0 if request.form['Contract'] == 'Month-to-month' else 
                         (1 if request.form['Contract'] == 'One year' else 2)],
            'PaperlessBilling': [1 if request.form['PaperlessBilling'] == 'Yes' else 0],
            'PaymentMethod': [2 if request.form['PaymentMethod'] == 'Electronic check' else 
                              (3 if request.form['PaymentMethod'] == 'Mailed check' else 
                               (0 if request.form['PaymentMethod'] == 'Bank transfer (automatic)' else 1))],
            'MonthlyCharges': [float(request.form['MonthlyCharges'])],
            'TotalCharges': [float(request.form['TotalCharges'])]
        }

        input_df = pd.DataFrame(input_data)
        input_df['TenureGroup'] = input_df['tenure'].apply(tenure_group)

        categorical_cols = input_df.select_dtypes(include=['category', 'object']).columns
        encoded_data = onehot_encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=onehot_encoder.get_feature_names_out(categorical_cols),
            index=input_df.index
        )

        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df, encoded_df], axis=1)

        input_df = scaler.transform(input_df)
        prediction = model.predict(input_df)
        result = 'This customer is likely to churn.' if prediction == 1 else 'This customer is not likely to churn.'

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)