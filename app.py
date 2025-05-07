import streamlit as st
from scripts.predict_risk import predict_stroke
from scripts.pose_tracking import collect_landmarks
from scripts.recovery_estimator import predict_recovery
from scripts.exercise_recommender import recommend_exercise

st.title("Stroke Recovery Prediction System")

menu = st.sidebar.selectbox("Select Option", ["Stroke Risk Prediction", "Recovery Monitoring", "Exercise Suggestion"])

if menu == "Stroke Risk Prediction":
    age = st.number_input("Age", 0, 100, 50)
    glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    if st.button("Predict Stroke Risk"):
        risk = predict_stroke(age, glucose, hypertension, heart_disease, bmi)
        st.success(f"Predicted Stroke Risk: {risk}%")

elif menu == "Recovery Monitoring":
    st.info("Click below to start live pose tracking")
    if st.button("Start Webcam"):
        session_path = 'data/landmark_sequences/session_1.npy'  # Define path for the session
        collect_landmarks(session_path)
        landmarks = np.load(session_path)
        recovery_score = predict_recovery(landmarks)
        st.success(f"Predicted Recovery: {recovery_score * 100:.2f}%")

elif menu == "Exercise Suggestion":
    recovery_score = st.slider("Enter Recovery Score", 0, 100, 50)
    suggestion = recommend_exercise(recovery_score)
    st.success(f"Suggested Exercise: {suggestion}")
