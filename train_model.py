import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (make sure heart.csv is in the same folder)
df = pd.read_csv(r"C:\Users\Student\Downloads\heart.csv")

# Prepare the data
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as model.pkl")
