import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("adr_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


# Function to take user input
def get_user_input():
    print("\nEnter the following patient details:")

    user_data = {
        "drug_name": input("Drug Name: "),
        "age": int(input("Age: ")),
        "sex": input("Sex (M/F): "),
        "weight": input("Weight (e.g., 16kg): "),
        "disease_status": input("Disease Status: "),
        "dosage": input("Dosage (e.g., 160mg): "),
        "dose_duration": input("Dose Duration (e.g., 6 months): "),
        "comorbidities": input("Comorbidities (if any, else 'None'): "),
        "lifestyle_factors": input("Lifestyle Factors (e.g., smoker, None): "),
        "pregnancy": input("Pregnancy (none if not applicable): "),
        "ast(10-40)": int(input("AST Value: ")),
        "alt(5-30)": int(input("ALT Value: ")),
        "alp(150-280)": int(input("ALP Value: "))
    }

    # Convert weight (e.g., "32kg") to a number
    user_data["weight"] = float(user_data["weight"].replace("kg", ""))

    # Convert categorical inputs using encoders
    for col in user_data:
        if col in label_encoders:
            user_data[col] = label_encoders[col].transform([user_data[col]])[0]

    # Convert input to model format
    input_array = np.array(list(user_data.values())).reshape(1, -1)

    return input_array


# Predict ADR, Symptoms, and Suggestions
def predict_adr():
    user_input = get_user_input()
    prediction = model.predict(user_input)

    print("\n--- Predicted Results ---")
    print(f"Adverse Drug Reactions: {prediction[0][0]}")
    print(f"Symptoms: {prediction[0][1]}")
    print(f"Medical Suggestions: {prediction[0][2]}")


# Run the prediction script
if __name__ == "__main__":
    predict_adr()
