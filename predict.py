import joblib
import numpy as np

# Load model
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample patient data
sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])

sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
