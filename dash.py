import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px

# Load dataset
df = pd.read_csv("Data.csv")

# Load trained models
priority_model = pickle.load(open("priority_model.pkl", "rb"))
category_model = pickle.load(open("category_model.pkl", "rb"))

# Load encoders and vectorizer
le_assigned_to = pickle.load(open("le_assigned_to.pkl", "rb"))
le_status = pickle.load(open("le_status.pkl", "rb"))
le_skill = pickle.load(open("le_skill.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Streamlit UI setup
st.set_page_config("Task Intelligence Dashboard", layout="wide")
tab1, tab2 = st.tabs(["📥 Task Prediction", " "])

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

with tab1:
    st.header("📥 Intelligent Task Priority & Category Prediction")

    with st.form("task_input_form"):
        task_description = st.text_area("Task Description")

        col1, col2 = st.columns(2)
        with col1:
            created_at = st.date_input("Created At")
        with col2:
            due_date = st.date_input("Due Date")

        col3, col4 = st.columns(2)
        with col3:
            assigned_to = st.selectbox("Assigned To", df['assigned_to'].unique())
        with col4:
            status = st.selectbox("Status", df['status'].unique())

        col5, col6 = st.columns(2)
        with col5:
            estimated_hours = st.number_input("Estimated Hours", min_value=0.0)
        with col6:
            actual_hours = st.number_input("Actual Hours", min_value=0.0)

        col7, col8 = st.columns(2)
        with col7:
            user_workload = st.number_input("User Workload", min_value=0.0)
        with col8:
            user_skill_level = st.selectbox("User Skill Level", df['user_skill_level'].unique())

        submitted = st.form_submit_button("Predict Task Attributes")

        if submitted:
            # Feature Engineering
            text_vector = vectorizer.transform([task_description])
            assigned_encoded = le_assigned_to.transform([assigned_to])[0]
            status_encoded = le_status.transform([status])[0]
            skill_encoded = le_skill.transform([user_skill_level])[0]
            days_to_due = (due_date - created_at).days

            input_features = np.hstack((
                text_vector.toarray()[0],
                [assigned_encoded, status_encoded, estimated_hours, actual_hours, user_workload, skill_encoded, days_to_due]
            )).reshape(1, -1)

            # Make predictions
            priority_pred = priority_model.predict(input_features)[0]
            category_pred = category_model.predict(input_features)[0]

            # Display side-by-side results
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.success(f"🟩 **Predicted Priority:** {priority_pred}")
            with col_result2:
                st.info(f"📂 **Predicted Category:** {category_pred}")

            # Store in history
            st.session_state.prediction_history.append({
                "Task Description": task_description,
                "Created At": created_at,
                "Due Date": due_date,
                "Assigned To": assigned_to,
                "Status": status,
                "Estimated Hours": estimated_hours,
                "Actual Hours": actual_hours,
                "User Workload": user_workload,
                "User Skill Level": user_skill_level,
                "Predicted Priority": priority_pred,
                "Predicted Category": category_pred
            })

    # Show prediction history table
    if st.session_state.prediction_history:
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)

