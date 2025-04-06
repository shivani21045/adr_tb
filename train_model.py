import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load the dataset
df = pd.read_csv("TB_PATIENT ADR DATASET.csv")

# Clean weight
df["weight"] = df["weight"].str.replace("kg", "", regex=True).astype(float)

# Fill missing values
df.fillna("Unknown", inplace=True)

# 🔄 Encode categorical columns using LabelEncoder
label_encoders = {}
for col in ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 💾 Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Combine text fields for TF-IDF
text_columns = ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]
df["combined_text"] = df[text_columns].astype(str).apply(lambda x: " ".join(x), axis=1)

# Set target columns
target_columns = ["adr", "symptoms", "suggestions", "pharmacokinetics", "pharmacodynamics", "drug_interactions"]
X = df["combined_text"]
y = df[target_columns]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)
joblib.dump(vectorizer, "text_vectorizer.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "adr_prediction_model.pkl")

# Save the cleaned dataset
df.to_csv("refined_dataset.csv", index=False)

print("✅ Model trained with text-based features and saved successfully!")
speak("Model Trained!")
