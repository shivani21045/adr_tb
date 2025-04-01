import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("TB_PATIENT ADR DATASET.csv")

df["weight"] = df["weight"].str.replace("kg", "", regex=True).astype(float)

df.fillna("Unknown", inplace=True)

text_columns = ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]

df["combined_text"] = df[text_columns].apply(lambda x: " ".join(x), axis=1)

target_columns = ["adr", "symptoms", "suggestions", "pharmacokinetics", "pharmacodynamics", "drug_interactions"]
X = df["combined_text"]
y = df[target_columns]

vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

joblib.dump(vectorizer, "text_vectorizer.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "adr_prediction_model.pkl")

print("✅ Model trained with text-based features and saved successfully!")
