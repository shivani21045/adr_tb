import joblib
import numpy as np
import pyttsx3
import pandas as pd
from rich.console import Console
from rich.table import Table
from colorama import Fore, Style, init
# Initialize Colorama
init(autoreset=True)

# Load the trained model and encoders
model = joblib.load("adr_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize text-to-speech engine
engine = pyttsx3.init()
console = Console()

# Load or create dataset file
dataset_file = "refined_dataset.csv"
try:
    existing_data = pd.read_csv(dataset_file)
except FileNotFoundError:
    existing_data = pd.DataFrame()

# Function to make the system speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to get safe user input with validation
def safe_input(prompt, input_type=str, allowed_values=None, multiple=False):
    while True:
        user_input = input(Fore.YELLOW + prompt).strip()  # Remove extra spaces
        if multiple:
            values = [val.strip() for val in user_input.split(",")]  # Split by comma and clean spaces
            if allowed_values:
                invalid_values = [val for val in values if val.lower() not in [x.lower() for x in allowed_values]]
                if invalid_values:
                    print(
                        Fore.RED + f"❌ Invalid choice(s): {', '.join(invalid_values)}. Allowed values: {', '.join(allowed_values)}")
                    continue  # Ask again if input is invalid
            return values  # Return a list of values

        if input_type == int:  # Handle integer input
            if user_input.isdigit():
                return int(user_input)
            else:
                print(Fore.RED + "❌ Invalid input! Please enter a valid number.")
        elif input_type == float:  # Handle float input
            try:
                return float(user_input.replace("kg", "").strip())  # Remove "kg" if exists
            except ValueError:
                print(Fore.RED + "❌ Invalid input! Please enter a valid number.")
        else:  # Handle text input
            if allowed_values and user_input.lower() not in [val.lower() for val in allowed_values]:
                print(Fore.RED + f"❌ Invalid choice! Allowed values: {', '.join(allowed_values)}")
            else:
                return user_input

# Function to take user input
def get_user_input():
    print(Fore.CYAN + "\n📋 Please enter the following patient details:\n")

    user_data = {
        "drug_name": safe_input("💊 Drug Name(s) (comma-separated): ", multiple=True),
        "age": safe_input("🎂 Age: ", int),
        "sex": safe_input("⚧️ Sex (M/F): ", allowed_values=["M", "F"]),
        "weight": safe_input("⚖️ Weight (e.g., 72kg): ", float),
        "disease_status": safe_input("🦠 Disease Status: "),
        "dosage": safe_input("💉 Dosage(s) (comma-separated): ", multiple=True),  # Allow multiple dosages
        "dose_duration": safe_input("⏳ Dose Duration: "),
        "comorbidities": safe_input("🩺 Comorbidities (comma-separated, if any, else 'None'): ", multiple=True),
        "lifestyle_factors": safe_input("🏋️ Lifestyle Factors (comma-separated, e.g., smoker, None): ", multiple=True),
        "pregnancy": safe_input("🤰 Pregnancy (yes/none): ", allowed_values=["yes", "none"]),
        "pregnancy_month": safe_input("📆 Pregnancy Month (if applicable, else 0): ", int),
        "ast(10-40)": safe_input("🩸 AST Value (10-40): ", int),
        "alt(5-30)": safe_input("🩸 ALT Value (5-30): ", int),
        "alp(150-280)": safe_input("🩸 ALP Value (150-280): ", int),
        "genetic_factors": safe_input("🧬 Genetic Factors (comma-separated, if any, else 'None'): ", multiple=True),
        "concomitant_medicine": safe_input("💊 Concomitant Medicines (comma-separated, if any, else 'None'): ", multiple=True)
    }

    # Convert categorical inputs using encoders
    # Convert categorical inputs using encoders
    for col in user_data:
        if col in label_encoders:
            try:
                if isinstance(user_data[col], list):  # If it's a list, encode each value separately
                    encoded_values = []
                    for val in user_data[col]:
                        if val in label_encoders[col].classes_:  # Check if value exists in encoder
                            encoded_values.append(label_encoders[col].transform([val])[0])
                        else:
                            print(
                                Fore.RED + f"❌ Invalid input! '{val}' is not recognized for {col}. Please enter a valid value.")
                            return get_user_input()  # Retry input
                    user_data[col] = encoded_values  # Store the encoded list
                else:
                    if user_data[col] in label_encoders[col].classes_:
                        user_data[col] = label_encoders[col].transform([user_data[col]])[0]
                    else:
                        print(
                            Fore.RED + f"❌ Invalid input! '{user_data[col]}' is not recognized. Please enter a valid value.")
                        return get_user_input()  # Retry input
            except ValueError:
                print(Fore.RED + f"❌ Unexpected error encoding '{user_data[col]}'. Please enter a valid value.")
                return get_user_input()  # Retry input

    # Convert input to model format (flatten lists if necessary)
    input_array = np.array([item if isinstance(item, (int, float)) else str(item) for item in user_data.values()]).reshape(1, -1)

    return user_data, input_array

# Predict ADR, Symptoms, Suggestions, Pharmacokinetics, Pharmacodynamics, Drug Interactions
def predict_adr():
    user_data, _ = get_user_input()  # Get raw user input

    # Load the trained text vectorizer
    vectorizer = joblib.load("text_vectorizer.pkl")

    # Combine user input fields into one text string
    text_columns = ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]
    # Combine user input fields into one text string
    text_columns = ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]
    combined_text = " ".join(
        str(user_data[col]) if isinstance(user_data[col], (str, int, float, np.int64, np.float64))
        else " ".join(map(str, user_data[col]))
        for col in text_columns
    )
    # Transform input text using the trained vectorizer
    user_input_transformed = vectorizer.transform([combined_text])

    # Load the trained model
    model = joblib.load("adr_prediction_model.pkl")

    # Predict ADRs
    predictions = model.predict(user_input_transformed)

    # Extract prediction results
    adr_results = predictions[0][0]

    if isinstance(adr_results, str):
        adr_text = adr_results
    elif isinstance(adr_results, (list, np.ndarray)):
        adr_text = ", ".join(adr_results)
    else:
        adr_text = "None"

    # Display results
    table = Table(title="🔬 **PREDICTION RESULTS**", title_style="bold magenta")
    table.add_column("🩸 Parameter", justify="left", style="cyan", no_wrap=True)
    table.add_column("📊 Prediction", justify="center", style="bold yellow")

    rows = [
        ("Adverse Drug Reactions", adr_text),
        ("Symptoms", predictions[0][1]),
        ("Medical Suggestions", predictions[0][2]),
        ("Pharmacokinetics", predictions[0][3]),
        ("Pharmacodynamics", predictions[0][4]),
        ("Drug Interactions", predictions[0][5]),
    ]

    for param, value in rows:
        table.add_row(param, value)
        table.add_section()

    console.print("\n")
    console.print(table)
    console.print("\n")

    speak(f"Detected Adverse Drug Reactions: {adr_text}")
    for param, value in rows:
        speak(f"{param}: {value}")

    print(Fore.RED + Style.BRIGHT + f"⚠️ WARNING: Detected ADRs: {adr_text}")

# Function to save data to dataset
# def save_data(user_data, prediction):
#     global existing_data
#
#     # Add prediction results to user data
#     user_data["adr"] = prediction[0][0]
#     user_data["symptoms"] = prediction[0][1]
#     user_data["suggestions"] = prediction[0][2]
#     user_data["pharmacokinetics"] = prediction[0][3]
#     user_data["pharmacodynamics"] = prediction[0][4]
#     user_data["drug_interactions"] = prediction[0][5]
#
#     # Convert to DataFrame and append to existing data
#     new_data = pd.DataFrame([user_data])
#     existing_data = pd.concat([existing_data, new_data], ignore_index=True)
#
#     # Save to CSV
#     existing_data.to_csv(dataset_file, index=False)
#     print(Fore.CYAN + "📁 Data saved successfully! The dataset has been updated.\n")

# Loop to allow multiple predictions
while True:
    predict_adr()
    another = input(Fore.MAGENTA + "\n🔄 Do you want to predict another ADR? (yes/no): ").strip().lower()
    if another != "yes":
        print(Fore.GREEN + "👋 Exiting... Thank you!")
        break  # Stop the loop if user enters anything other than "yes"
