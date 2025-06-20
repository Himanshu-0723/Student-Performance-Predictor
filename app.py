import streamlit as st
import pandas as pd
import numpy as np
import joblib

preprocessor = joblib.load("saved_models/preprocessor.pkl")
models = {
    "Logistic Regression": joblib.load("saved_models/logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("saved_models/decision_tree_model.pkl"),
    "Random Forest": joblib.load("saved_models/random_forest_model.pkl"),
    "SVM": joblib.load("saved_models/svm_model.pkl"),
    "KNN": joblib.load("saved_models/knn_model.pkl")
}

st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will pass or fail based on various features.")


with st.form("input_form"):
    st.header("Enter Student Information")

    reason = st.selectbox("Reason for choosing this school", ['home', 'reputation', 'course', 'other'])
    traveltime = st.slider("Travel time (1= <15 min ... 4= >1 hour)", 1, 4, 1)
    studytime = st.slider("Weekly study time (1= <2 hrs ... 4= >10 hrs)", 1, 4, 2)
    failures = st.slider("Past class failures", 0, 4, 0)

    schoolsup = st.selectbox("School support", ['yes', 'no'])
    famsup = st.selectbox("Family support", ['yes', 'no'])
    paid = st.selectbox("Extra paid classes", ['yes', 'no'])
    activities = st.selectbox("Extracurricular activities", ['yes', 'no'])
    nursery = st.selectbox("Attended nursery school", ['yes', 'no'])
    higher = st.selectbox("Wants higher education", ['yes', 'no'])
    internet = st.selectbox("Internet access at home", ['yes', 'no'])
    romantic = st.selectbox("In a romantic relationship", ['yes', 'no'])

    famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
    freetime = st.slider("Free time after school (1–5)", 1, 5, 3)
    goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
    Dalc = st.slider("Workday alcohol consumption (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend alcohol consumption (1–5)", 1, 5, 2)
    health = st.slider("Current health status (1–5)", 1, 5, 3)
    absences = st.number_input("Number of absences", min_value=0, value=4)

    selected_model = st.selectbox("Choose a classification model:", list(models.keys()))

    submitted = st.form_submit_button("Predict")

if submitted:
    def binary(val): return 1 if val == 'yes' else 0

    input_dict = {
        'reason': reason,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': binary(schoolsup),
        'famsup': binary(famsup),
        'paid': binary(paid),
        'activities': binary(activities),
        'nursery': binary(nursery),
        'higher': binary(higher),
        'internet': binary(internet),
        'romantic': binary(romantic),
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences
    }

    input_df = pd.DataFrame([input_dict])

    input_processed = preprocessor.transform(input_df)
    model = models[selected_model]
    prediction = model.predict(input_processed)[0]
    proba = model.predict_proba(input_processed)[0][1] if hasattr(model, 'predict_proba') else None

    st.subheader("📊 Prediction Result:")
    st.write("🎯 **PASS**" if prediction == 1 else "❌ **FAIL**")
    if proba is not None:
        st.write(f"📈 **Probability of Passing**: {proba * 100:.2f}%")
