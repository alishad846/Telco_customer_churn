import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model once when the application starts
model = pickle.load(open("model.sav", "rb"))

# Load the dataset once when the application starts
df_1 = pd.read_csv("first_telc.csv")


@app.route("/", methods=['GET'])
def loadPage():
    return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    try:
        # Retrieve form data
        form_data = [request.form.get(key) for key in [
            'query1', 'query2', 'query3', 'query4', 'query5', 'query6', 'query7',
            'query8', 'query9', 'query10', 'query11', 'query12', 'query13', 'query14',
            'query15', 'query16', 'query17', 'query18', 'query19'
        ]]

        # Check if any field is empty
        if any(field is None or field.strip() == '' for field in form_data):
            return render_template('home.html', output1="Error: All fields are required.", output2="Please fill in all the fields.",
                                   **{f'query{i + 1}': form_data[i] for i in range(19)})

        # Cast relevant columns to correct types with error handling
        try:
            new_data = {
                'SeniorCitizen': int(form_data[0]),
                'MonthlyCharges': float(form_data[1]),
                'TotalCharges': float(form_data[2]),
                'gender': form_data[3],
                'Partner': form_data[4],
                'Dependents': form_data[5],
                'PhoneService': form_data[6],
                'MultipleLines': form_data[7],
                'InternetService': form_data[8],
                'OnlineSecurity': form_data[9],
                'OnlineBackup': form_data[10],
                'DeviceProtection': form_data[11],
                'TechSupport': form_data[12],
                'StreamingTV': form_data[13],
                'StreamingMovies': form_data[14],
                'Contract': form_data[15],
                'PaperlessBilling': form_data[16],
                'PaymentMethod': form_data[17],
                'tenure': int(form_data[18])
            }
        except ValueError as ve:
            return render_template('home.html', output1="Error: Invalid input type.", output2=str(ve),
                                   **{f'query{i + 1}': form_data[i] for i in range(19)})

        # Create a DataFrame with the form data
        new_df = pd.DataFrame([new_data])

        # Combine with existing data
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2['tenure'], bins=range(1, 80, 12), right=False, labels=labels)

        # Drop the original 'tenure' column
        df_2.drop(columns=['tenure'], axis=1, inplace=True)

        # Convert categorical variables to dummy variables
        df_2_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

        # Ensure that the new data has all columns expected by the model
        df_2_dummies = df_2_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict using the model
        new_data_dummies = df_2_dummies.tail(1)
        prediction = model.predict(new_data_dummies)
        probability = model.predict_proba(new_data_dummies)[:, 1]

        # Determine the output messages
        if prediction == 1:
            o1 = "This customer is likely to churn!"
            o2 = f"Confidence: {probability[0] * 100:.2f}%"
        else:
            o1 = "This customer is likely to continue!"
            o2 = f"Confidence: {probability[0] * 100:.2f}%"

        return render_template('home.html',
                               output1=o1,
                               output2=o2,
                               **{f'query{i + 1}': form_data[i] for i in range(19)})

    except Exception as e:
        return render_template('home.html', output1="An unexpected error occurred.", output2=str(e),
                               **{f'query{i + 1}': request.form.get(f'query{i + 1}', '') for i in range(19)})


if __name__ == '__main__':
    app.run(debug=True)
