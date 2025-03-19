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
def safe_input(prompt, input_type=str, allowed_values=None):
    while True:
        user_input = input(Fore.YELLOW + prompt).strip()  # Remove extra spaces
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
        "drug_name": safe_input("💊 Drug Name: "),
        "age": safe_input("🎂 Age: ", int),
        "sex": safe_input("⚧️ Sex (M/F): ", allowed_values=["M", "F"]),
        "weight": safe_input("⚖️ Weight (e.g., 72kg): ", float),
        "disease_status": safe_input("🦠 Disease Status: "),
        "dosage": safe_input("💉 Dosage (mg): "),
        "dose_duration": safe_input("⏳ Dose Duration (e.g., 6 months): "),
        "comorbidities": safe_input("🩺 Comorbidities (if any, else 'None'): "),
        "lifestyle_factors": safe_input("🏋️ Lifestyle Factors (e.g., smoker, None): "),
        "pregnancy": safe_input("🤰 Pregnancy (yes/no): ", allowed_values=["yes", "no","none"]),
        "pregnancy_month": safe_input("📆 Pregnancy Month (if applicable, else 0): ", int),
        "ast(10-40)": safe_input("🩸 AST Value (10-40): ", int),
        "alt(5-30)": safe_input("🩸 ALT Value (5-30): ", int),
        "alp(150-280)": safe_input("🩸 ALP Value (150-280): ", int),
        "genetic_factors": safe_input("🧬 Genetic Factors (if any, else 'None'): "),
        "concomitant_medicine": safe_input("💊 Concomitant Medicines (if any, else 'None'): ")
    }

    # Convert categorical inputs using encoders
    for col in user_data:
        if col in label_encoders:
            try:
                user_data[col] = label_encoders[col].transform([user_data[col]])[0]
            except ValueError:
                print(Fore.RED + f"❌ Invalid input! '{user_data[col]}' is not recognized. Please enter a valid value.")
                return get_user_input()  # Retry input

    # Convert input to model format
    input_array = np.array(list(user_data.values())).reshape(1, -1)

    return user_data, input_array

# Predict ADR, Symptoms, Suggestions, Pharmacokinetics, Pharmacodynamics, Drug Interactions
def predict_adr():
    user_data, user_input = get_user_input()
    prediction = model.predict(user_input)

    # Extract prediction results
    adr_result = prediction[0][0]  # ADR result
    drug_interaction_result = f"{prediction[0][5]} (if any)"  # Add "(if any)" to drug interactions

    # Create a nice-looking table using rich
    table = Table(title="🔬 **PREDICTION RESULTS**", title_style="bold magenta")

    table.add_column("🩸 Parameter", justify="left", style="cyan", no_wrap=True)
    table.add_column("📊 Prediction", justify="center", style="bold yellow")

    table.add_row("Adverse Drug Reactions", f"{prediction[0][0]}")
    table.add_row("Symptoms", f"{prediction[0][1]}")
    table.add_row("Medical Suggestions", f"{prediction[0][2]}")
    table.add_row("Pharmacokinetics", f"{prediction[0][3]}")
    table.add_row("Pharmacodynamics", f"{prediction[0][4]}")
    table.add_row("Drug Interactions", drug_interaction_result)

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Voice output based on ADR result
    if adr_result.lower() == "none":
        speak("Adverse Drug Reaction is NOT detected! The drug is likely safe.")
        print(Fore.GREEN + Style.BRIGHT + "✅ ADR is NOT detected! The drug is likely safe.")
    else:
        speak("WARNING: Adverse Drug Reaction detected!.")
        print(Fore.RED + Style.BRIGHT + "⚠️ WARNING: ADR detected!.")

    # Save input + output to dataset
    save_data(user_data, prediction)

# Function to save data to dataset
def save_data(user_data, prediction):
    global existing_data

    # Add prediction results to user data
    user_data["adr"] = prediction[0][0]
    user_data["symptoms"] = prediction[0][1]
    user_data["suggestions"] = prediction[0][2]
    user_data["pharmacokinetics"] = prediction[0][3]
    user_data["pharmacodynamics"] = prediction[0][4]
    user_data["drug_interactions"] = prediction[0][5]

    # Convert to DataFrame and append to existing data
    new_data = pd.DataFrame([user_data])
    existing_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Save to CSV
    existing_data.to_csv(dataset_file, index=False)
    print(Fore.CYAN + "📁 Data saved successfully! The dataset has been updated.\n")

# Loop to allow multiple predictions
while True:
    predict_adr()
    another = input(Fore.MAGENTA + "\n🔄 Do you want to predict another ADR? (yes/no): ").strip().lower()
    if another != "yes":
        print(Fore.GREEN + "👋 Exiting... Thank you!")
        break  # Stop the loop if user enters anything other than "yes"
