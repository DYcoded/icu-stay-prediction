import streamlit as st
import joblib
import pandas as pd
import numpy as np



st.set_page_config(page_title="Hospital AI Prediction", page_icon="🏥", layout="centered")


@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('hospital_model.pkl')
        cols = joblib.load('model_columns.pkl')
        return model, cols
    except FileNotFoundError:
        return None, None

model, model_columns = load_model_resources()


st.title("🏥 Hospital Length of Stay Prediction")
st.markdown("""
This AI-powered system predicts whether a patient will have a **Short Stay (0-10 days)** or a **Long Stay (>10 days)** based on clinical and demographic parameters.
""")

if model is None:
    st.error("ERROR: Model files ('hospital_model.pkl' or 'model_columns.pkl') not found!")
    st.info("Please ensure these files are in the same directory as app.py.")
    st.stop()

st.divider()


col1, col2 = st.columns(2)

with col1:
    st.subheader("🩺 Clinical Vitals")
    
    bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 240, 120, help="Standard range is around 120.")
    pulse = st.number_input("Heart Rate (bpm)", 40, 200, 80)
    
    severity_val = 1 
    severity_text = "Stable (Minor)"
    
    if bp > 160 or pulse > 110:
        severity_val = 0 
        severity_text = "🚨 CRITICAL (Extreme)"
        st.error(f"Alert: {severity_text}")
    elif bp > 140 or pulse > 100:
        severity_val = 2 
        severity_text = "⚠️ Elevated (Moderate)"
        st.warning(f"Warning: {severity_text}")
    else:
        st.success(f"Status: {severity_text}")


    adm_type = st.selectbox("Admission Type", ["Emergency", "Trauma", "Urgent"])
    
    
    adm_map = {"Emergency": 0, "Trauma": 1, "Urgent": 2}
    adm_val = adm_map[adm_type]

with col2:
    st.subheader("👤 Patient Details")
    
    
    dept = st.selectbox("Department", 
                        ["Anesthesia", "Gynecology", "Radiotherapy", "Surgery", "TB & Chest Disease"])
    
    
    dept_map = {"Anesthesia": 0, "Gynecology": 1, "Radiotherapy": 2, "Surgery": 3, "TB & Chest Disease": 4}
    dept_val = dept_map[dept]
    
    
    age_code = st.selectbox("Age Group", range(10), format_func=lambda x: f"{x} ({x*10}-{x*10+10} Years)")
    deposit = st.number_input("Admission Deposit ($)", 0, 50000, 4000, step=500)
    visitors = st.slider("Visitors with Patient", 0, 10, 2)

st.divider()
predict_btn = st.button("🔍 Analyze & Predict", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner('AI is analyzing clinical data...'):
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0 
        
        try:
            input_data['Admission_Deposit'] = deposit
            input_data['Visitors with Patient'] = visitors
            input_data['Age'] = age_code
            
            if 'Type of Admission' in input_data.columns:
                input_data['Type of Admission'] = adm_val
            if 'Department' in input_data.columns:
                input_data['Department'] = dept_val
            if 'Severity of Illness' in input_data.columns:
                input_data['Severity of Illness'] = severity_val
            
            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]

            if prediction == 0 and severity_val == 0:
                st.warning("⚠️ CLINICAL OVERRIDE: Although the AI predicts a 'Short Stay', the patient's vital signs are CRITICAL. Intensive Care protocols should take precedence over the model's statistical prediction.")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediction == 0:
                    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100) 
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100) 

            with col_res2:
                if prediction == 0:
                    st.success("✅ RESULT: SHORT STAY (0-10 Days)")
                    st.progress(int(probs[0]*100))
                    st.caption(f"Confidence Score: {probs[0]*100:.1f}%")
                    st.info("💡 Insight: Patient condition appears stable. Early discharge expected.")
                else:
                    st.error("🚨 RESULT: LONG STAY (>10 Days)")
                    st.progress(int(probs[1]*100))
                    st.caption(f"Confidence Score: {probs[1]*100:.1f}%")
                    
                    reasons = []
                    if dept == "Surgery": reasons.append("Surgery typically implies a post-op recovery period.")
                    if dept == "TB & Chest Disease": reasons.append("Chest diseases often require prolonged observation.")
                    if adm_type == "Trauma": reasons.append("Trauma cases statistically have longer recovery times.")
                    if severity_val == 0: reasons.append("Critical vitals suggest a need for intensive care.")
                    if age_code >= 7: reasons.append("Advanced age may affect recovery speed.")
                    
                    if reasons:
                        st.warning("AI Reasoning:\n" + "\n".join([f"- {r}" for r in reasons]))
                    else:
                        st.warning("Insight: Resource planning recommended for long-term stay.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.code("Mismatch between model columns and input data. Check 'model_columns.pkl'.")