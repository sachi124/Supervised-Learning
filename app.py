import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Loading the saved model and metadata
with open("model.pkl", "rb") as f:
  model = pickle.load(f)

with open("X_train_columns.pkl", "rb") as f:
  X_train_columns = pickle.load(f)

with open("dataframe_stats.pickle", "rb") as f:
  stats = pickle.load(f)
  glucose_mean = stats["glucose_mean"]


# Prediction Function
def predict_diabetes(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    HbA1c_level,
    blood_glucose_level,
):
  # Creating input dataframe
  patient = pd.DataFrame([{
      "gender": gender,
      "age": age,
      "hypertension": hypertension,
      "heart_disease": heart_disease,
      "smoking_history": smoking_history,
      "bmi": bmi,
      "HbA1c_level": HbA1c_level,
      "blood_glucose_level": blood_glucose_level,
  }])

  # Matching training columns using dummy variables and reindex
  patient = pd.get_dummies(
      patient, columns=["gender", "smoking_history"], drop_first=True
  )
  patient = patient.reindex(columns=X_train_columns, fill_value=0)

  # Model Prediction
  probability = float(model.predict_proba(patient)[0, 1])
  prediction = int(probability >= 0.40)

  results = [" Not Diabetic", " Diabetic"]
  advice = [
      (
          "The model predicts the non-diabetic class. Keep healthy habits and"
          " consult a doctor if you have any concerns."
      ),
      (
          "The model predicts the diabetic class. Please consult a doctor for"
          " proper clinical evaluation and confirmation."
      ),
  ]

  # Checking Status
  bmi_status = [" Outside healthy range", " Within healthy range"][
      18.5 <= bmi <= 24.9
  ]
  hba1c_status = [" Above normal range", " Within normal range"][
      HbA1c_level < 5.7
    ]
  glucose_status = [" Above dataset average", " At or below dataset average"][
      blood_glucose_level <= glucose_mean
  ]
  hypertension_status = [" Present", " None"][hypertension == 0]
  heart_status = [" Reported", " None reported"][heart_disease == 0]

  report = f"""### 🩺 Input Review
- **BMI:** {bmi:.2f} ({bmi_status})
- **HbA1c:** {HbA1c_level:.1f}% ({hba1c_status})
- **Blood Glucose:** {blood_glucose_level:.0f} mg/dL ({glucose_status})
- **Hypertension:** {hypertension} ({hypertension_status})
- **Heart Disease:** {heart_disease} ({heart_status})"""

  return f"""## {results[prediction]}

**Estimated diabetes probability:** {probability:.1%}

{advice[prediction]}

{report}

⚠️ *Note: This is a machine learning prediction, not a formal medical diagnosis.*"""


# Gradio Interface Layout
with gr.Blocks() as demo:
  gr.Markdown(
      "#  Diabetes Prediction Guide\n"
      "Enter the patient information below to get a risk evaluation."
  )

  with gr.Row():
    gender_input = gr.Dropdown(
        ["Female", "Male", "Others"], value="Female", label="Gender"
    )
    smoking_input = gr.Dropdown(
        ["never", "No Info", "current", "former", "ever", "not current"],
        value="never",
        label="Smoking History",
    )

  with gr.Row():
    age_input = gr.Number(
        value=40, minimum=1, maximum=100, step=1, label="Age"
    )
    hypertension_input = gr.Number(
        value=0,
        minimum=0,
        maximum=1,
        step=1,
        label="Hypertension",
        info="0 = No, 1 = Yes",
    )
    heart_disease_input = gr.Number(
        value=0,
        minimum=0,
        maximum=1,
        step=1,
        label="Heart Disease",
        info="0 = No, 1 = Yes",
    )

  # Fixed: This row is now correctly aligned at the main block level
  with gr.Row():
    bmi_input = gr.Number(value=24.5, label="BMI")
    hba1c_input = gr.Number(value=5.5, label="HbA1c Level")
    glucose_input = gr.Number(value=120.0, label="Blood Glucose Level")

  predict_button = gr.Button(
      " Predict Diabetes Status", variant="primary", size="lg"
  )
  result_output = gr.Markdown(
      "Enter the information and click the prediction button."
  )

  predict_button.click(
      fn=predict_diabetes,
      inputs=[
          gender_input,
          age_input,
          hypertension_input,
          heart_disease_input,
          smoking_input,
          bmi_input,
          hba1c_input,
          glucose_input,
      ],
      outputs=result_output,
  )

if __name__ == "__main__":
  demo.launch(
      theme=gr.themes.Soft(primary_hue="sky", secondary_hue="cyan"),
      share=True,
      show_error=True,
  )