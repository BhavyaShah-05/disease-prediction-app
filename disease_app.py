import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Expanded dataset
data = {
    'Symptom1': ['Fever', 'Cough', 'Headache', 'Fever', 'Nausea', 'Cough', 'Headache', 'Fever',
                 'Fatigue', 'Sneezing', 'Fever', 'Sore Throat', 'Fever', 'Cough', 'Body Pain', 'Loss of Taste'],
    'Symptom2': ['Cough', 'Fever', 'Nausea', 'Headache', 'Cough', 'Fatigue', 'Fever', 'Nausea',
                 'Body Pain', 'Sore Throat', 'Body Pain', 'Cough', 'Loss of Taste', 'Sore Throat', 'Fatigue', 'Fever'],
    'Disease': ['Flu', 'Flu', 'Migraine', 'Migraine', 'Food Poisoning', 'Cold', 'Cold', 'Food Poisoning',
                'Malaria', 'Allergy', 'Typhoid', 'Allergy', 'COVID-19', 'COVID-19', 'Malaria', 'COVID-19']
}

df = pd.DataFrame(data)

# Label encoding
le_symptom = LabelEncoder()
le_disease = LabelEncoder()

df['Symptom1_enc'] = le_symptom.fit_transform(df['Symptom1'])
df['Symptom2_enc'] = le_symptom.transform(df['Symptom2'])
df['Disease_enc'] = le_disease.fit_transform(df['Disease'])

X = df[['Symptom1_enc', 'Symptom2_enc']]
y = df['Disease_enc']

# Train two models
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
dt_model.fit(X, y)
rf_model.fit(X, y)

# UI
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🧬 Disease Prediction")
st.caption("Select symptoms and get a machine learning-based disease prediction with visual feedback.")

# Model switch
model_choice = st.radio("Select Model", ['Decision Tree', 'Random Forest'])

# Symptoms
symptom_list = list(le_symptom.classes_)
symptom1 = st.selectbox("Select First Symptom", symptom_list)
symptom2 = st.selectbox("Select Second Symptom", symptom_list)

# Predict Button
if st.button("🔍 Predict Disease"):
    input_encoded = [[
        le_symptom.transform([symptom1])[0],
        le_symptom.transform([symptom2])[0]
    ]]

    # Choose model
    model = dt_model if model_choice == 'Decision Tree' else rf_model
    pred_encoded = model.predict(input_encoded)[0]
    pred_probs = model.predict_proba(input_encoded)[0]
    predicted_disease = le_disease.inverse_transform([pred_encoded])[0]

    st.success(f"🩺 Based on your symptoms, you may have **{predicted_disease}**.")

    # Build prediction probabilities DataFrame
    prob_df = pd.DataFrame({
        'Disease': le_disease.inverse_transform(range(len(pred_probs))),
        'Probability': pred_probs
    }).sort_values(by="Probability", ascending=False)

    # Show top 5 predictions as a table
    st.subheader("🔝 Top Predicted Diseases")
    st.dataframe(prob_df.head(5).reset_index(drop=True))

    # Bar chart
    st.subheader("📊 Probability Bar Chart")
    fig1, ax1 = plt.subplots()
    ax1.barh(prob_df['Disease'], prob_df['Probability'], color='skyblue')
    ax1.set_xlabel("Probability")
    ax1.set_ylabel("Disease")
    ax1.set_title("Disease Prediction Confidence")
    st.pyplot(fig1)

    