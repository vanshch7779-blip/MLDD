import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# Load trained model
model = joblib.load('my_model.pkl')

def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):

    # Create input dataframe with correct feature order
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    # Make prediction directly using original input features
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        return f"⚠️ Heart Disease Detected\nConfidence: {probability[1]*100:.2f}%"
    else:
        return f"✅ No Heart Disease\nConfidence: {probability[0]*100:.2f}%"

# Build the dashboard
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age (years)", value=50),
        gr.Radio(choices=[0, 1], label="Sex (0=Female, 1=Male)", value=1),
        gr.Slider(0, 3, step=1, label="Chest Pain Type (cp)", value=1),
        gr.Number(label="Resting BP (trestbps)", value=120),
        gr.Number(label="Cholesterol (chol)", value=200),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 (fbs)", value=0),
        gr.Slider(0, 2, step=1, label="Rest ECG (restecg)", value=1),
        gr.Number(label="Max Heart Rate (thalach)", value=150),
        gr.Radio(choices=[0, 1], label="Exercise Angina (exang)", value=0),
        gr.Number(label="Oldpeak (ST depression)", value=1.0),
        gr.Slider(0, 2, step=1, label="Slope", value=1),
        gr.Slider(0, 4, step=1, label="Major Vessels (ca)", value=0),
        gr.Slider(0, 3, step=1, label="Thal", value=2),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=3),
    title="🫀 Heart Disease Prediction Dashboard",
    description="Enter patient details and click Submit to predict heart disease risk."
)


# Launch for Render (uses PORT environment variable)
# Launch for Render (uses PORT environment variable)
demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
