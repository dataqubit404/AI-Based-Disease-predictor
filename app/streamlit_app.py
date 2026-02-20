import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ========================
# 🎨 Theme Handler
# ========================
def apply_theme():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    choice = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

    if choice == "Dark":
        st.session_state.dark_mode = True
        st.markdown(
            """
            <style>
            body, .stApp {
                background-color: #121212 !important;
                color: #ffffff !important;
            }
            .stSidebar, .css-1d391kg, .css-qrbaxs {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            .stButton>button {
                background-color: #bb86fc !important;
                color: black !important;
                border-radius: 8px !important;
                font-weight: bold;
            }
            .stNumberInput>div>input, .stSelectbox>div>select, .stSlider>div>div>input {
                background-color: #333 !important;
                color: #ffffff !important;
                font-weight: bold;
            }
            h1, h2, h3, h4, h5, h6, label, .css-10trblm {
                color: #ffffff !important;
                text-shadow: 0px 0px 6px rgba(255,255,255,0.8);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body, .stApp {
                background-color: white !important;
                color: black !important;
            }

            /* Fix tooltip icon color in light mode */
            [data-testid="stWidgetTooltipIcon"] svg path,
            [data-testid="stWidgetTooltipIcon"] svg circle,
            [data-testid="stWidgetTooltipIcon"] svg line {
                stroke: #6C63FF !important;
                fill: #6C63FF !important;
                color: #6C63FF !important;
            }



            .stAppHeader, .stHeader {
                background: #1e1e1e !important; 
                color: #222 !important;         
            }

            /* Sidebar background */
            .stSidebar, .css-1d391kg, .css-qrbaxs {
                background-color: #1e1e1e !important; 
                color: black !important;              
            }
            
            /* Sidebar titles and labels */
            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
            .stSidebar label, .stSidebar .css-10trblm, .stSidebar .stRadio label, .stSidebar .stMarkdown,
            .stSidebar .stTitle, .stSidebar .stButton > button {
                color: white !important;
                text-shadow: none !important;
            }

            .stButton>button {
                background-color: #bb86fc !important;
                color: white !important;
                border-radius: 8px !important;
                font-weight: bold;
            }

            .stNumberInput>div>input, 
            .stSelectbox>div>div , 
            .stSelectbox>div>select, 
            .stSlider>div>div>input,
            .stSlider .st-bf {
                background-color: #ffffff !important;
                color: black !important;
                font-weight: normal;
                border: 1px solid #ccc !important;
            }
            /* Main app text */
            h1, h2, h3, h4, h5, h6, label, .css-10trblm, .st-ae, .st-ag, .st-ai, .st-as {
                color: black !important;
                text-shadow: none !important;
            }

            /* Dropdown box*/
            [data-baseweb="popover"], 
            [data-baseweb="menu"], 
            ul[data-baseweb="menu"], 
            div[data-baseweb="popover"] > div {
                background-color: #fff !important;
                color: #222 !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
            }

            [data-baseweb="menu"] [role="option"], 
            ul[data-baseweb="menu"] li, 
            li[role="option"] {
                background: #fff !important;
                color: #222 !important;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )


# ========================
# 🛠 Feature Alignment
# ========================
def prepare_input(input_dict, feature_file, scaler_file):
    features = joblib.load(feature_file)
    scaler = joblib.load(scaler_file)

    df_input = pd.DataFrame([input_dict])

    # Add missing columns
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Correct order
    df_input = df_input[features]

    return scaler.transform(df_input)

# ========================
# 📊 Prediction Display
# ========================
def show_prediction(prediction, probabilities, positive_label, negative_label):
    prob_positive = probabilities[1]
    prob_negative = probabilities[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{positive_label}")
        st.progress(prob_positive)
        st.write(f"{prob_positive*100:.1f}%")
    with col2:
        st.markdown(f"{negative_label}")
        st.progress(prob_negative)
        st.write(f"{prob_negative*100:.1f}%")

    if prediction == 1:
        st.markdown(
            f"<h3 style='color:#dc3545;'>Prediction: {positive_label}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h3 style='color:#28a745;'>Prediction: {negative_label}</h3>",
            unsafe_allow_html=True,
        )

# ========================
# 🚀 Main App
# ========================
apply_theme()

with st.sidebar:
    st.title("AI Disease Predictor")
    st.write("Choose a disease and enter your health data to get predictions.")
    st.write("Developed with ❤ using Streamlit")

disease = st.selectbox("Choose a Disease", ["Diabetes", "Heart Disease", "Parkinson's", "Kidney Disease"])

# ---------------- Diabetes ----------------
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    st.write("")
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age (20–80)", 20, 80, 30, help="Older age increases risk")
    bmi = st.slider("BMI (10–50)", 10.0, 50.0, 25.0, help="Healthy BMI: 18.5–24.9")
    if gender == "Female":
        pregnancies = st.slider("Pregnancies (0–20)", 0, 20, 0, help="Number of times pregnant")
    else:
        pregnancies = 0
    st.subheader("Test Results")
    glucose = st.slider("Glucose Fasting(40–200)", 40, 200, 120, help="Normal fasting glucose levels: less than 100 mg/dL")
    insulin = st.slider("Insulin (15–276)", 15, 276, 80, help="Insulin level. Normal: 16–166 μU/mL")
    blood_pressure = st.slider("Diastolic Blood Pressure (mmHg)", 40, 140, 70, help="Diastolic BP. Normal: <80 mmHg")
    st.subheader("Family History and other")
    diabetes_pedigree = st.slider("Diabetes Pedigree Function (0–2.5)", 0.0, 2.5, 0.5, help="Likelihood from family history")
    skin_thickness = st.slider("Skin Thickness (5–60)", 5, 60, 20, help="Indicator of body fat")

    if st.button("Predict Diabetes Risk"):
        try:
            model = joblib.load("models/diabetes.pkl")
            input_data = {
                "Age": age,
                "BMI": bmi,
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "Insulin": insulin,
                "BloodPressure": blood_pressure,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "SkinThickness": skin_thickness
            }
            input_scaled = prepare_input(input_data, "models/diabetes_features.pkl", "models/diabetes_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Diabetes Detected", "No Diabetes")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ---------------- Heart Disease ----------------
elif disease == "Heart Disease":
    st.header("🫀 Heart Disease Prediction")
    st.write("")
    st.subheader("Personal Information")
    sex = st.selectbox("Gender", ["Male", "Female"], help="Males usually higher risk")
    age = st.slider("Age", 18, 100, 45, help="Older age increases risk")
    st.subheader("Risk Factors")
    cp_labels = [
        "Typical Angina (chest pain from heart)",
        "Atypical Angina (unusual chest pain, less likely heart-related)",
        "Non-anginal Pain (chest pain not from heart)",
        "Asymptomatic (no chest pain, but heart issues present)"
    ]
    cp_raw = st.selectbox("Chest Pain Type", cp_labels,
        help="How would you describe your chest pain?")

    cp_map = {
        cp_labels[0]: 0,
        cp_labels[1]: 1,
        cp_labels[2]: 2,
        cp_labels[3]: 3
    }
    cp = cp_map[cp_raw]

    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120, help="Normal: <120 mmHg")
    chol = st.slider("Cholesterol", 100, 600, 240, help="Normal: less than 200 mg/dL")
    st.subheader("Blood Sugar and ECG")
    fbs_raw = st.selectbox("Is your Fasting Blood Sugar greater than 120 mg/dl", ["Yes", "No"], help="Select Yes if your test result is above 120 mg/dL; select No if it is 120 or below")
    fbs = 1 if fbs_raw == "Yes" else 0
    restecg_raw = st.selectbox(
        "Resting ECG Results", 
        ["Normal", "Abnormal ST-T Wave", "Left Ventricular Hypertrophy"], 
        help="Select the ECG result that applies")
    restecg_map = {
        "Normal": 0,
        "Abnormal ST-T Wave" : 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map[restecg_raw]
    st.subheader("Exercise and Symptoms")
    thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150, help="Higher capacity = healthier")
    exang_raw = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Select Yes if you have a Exercise Induced Angina; Select No if you do not have a Exercise Induced Angina")
    exang = 1 if exang_raw == "Yes" else 0
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, help="Exercise-induced depression")
    st.subheader("Other")
    slope_raw = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat (Horizontal)", "Downsloping"], help="Shape of the ST segment after exercise: Upsloping (rising), Flat (horizontal), Downsloping (falling)")
    slope_map = {
        "Upsloping": 0,
        "Flat (Horizontal)": 1,
        "Downsloping": 2
    }
    slope = slope_map[slope_raw]
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3], help="Higher = more blockage risk")
    thal_raw = st.selectbox("Thal", ["Normal", "Fixed defect", "Reversible defect"])
    thal_map = {
        "Normal": 0,
        "Fixed defect": 1,
        "Reversible defect": 2
    }
    thal = thal_map[thal_raw]

    if st.button("Predict Heart Disease"):
        try:
            model = joblib.load("models/heart.pkl")
            input_data = {
                "age": age,
                "sex": 1 if sex == "Male" else 0,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
            input_scaled = prepare_input(input_data, "models/heart_features.pkl", "models/heart_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Heart Disease Detected", "No Heart Disease")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ---------------- Parkinson's ----------------
elif disease == "Parkinson's":
    st.header("🧠 Parkinson's Prediction")
    st.write("")
    st.subheader("Personal Information")
    age = st.slider("Age", 18, 150, 35, help = "Your age")
    st.subheader("Voice Measurements")
    fo = st.slider("Average Vocal Fundamental Frequency (fo)", 100.0, 400.0, 150.0, help="Avg voice pitch")
    fhi = st.slider("Maximum Vocal Frequency (fhi)", 150.0, 500.0, 200.0, help="Max pitch")
    flo = st.slider("Minimum Vocal Frequency (flo)", 50.0, 200.0, 100.0, help="Min pitch")
    st.subheader("Voice Quality")
    jitter = st.slider("Jitter (%)", 0.0, 1.0, 0.01, 0.001, help="Voice frequency variation")
    shimmer = st.slider("Shimmer", 0.0, 1.0, 0.02, 0.001, help="Voice amplitude variation")
    rpde = st.slider("RPDE", 0.0, 1.0, 0.5, 0.01, help="Voice disorder measure")
    dfa = st.slider("DFA", 0.0, 1.0, 0.5, 0.01, help="Signal fractal scaling")
    spread1 = st.slider("Spread1", -10.0, 0.0, -5.0, 0.1, help="Voice frequency spread")
    spread2 = st.slider("Spread2", -5.0, 5.0, 0.0, 0.1, help="Voice stability")
    d2 = st.slider("D2", 1.0, 5.0, 2.0, 0.1, help="Signal complexity")
    PPE = st.slider("PPE", 0.0, 1.0, 0.1, 0.01, help="Pitch Period Entropy")

    if st.button("Predict Parkinson's"):
        try:
            model = joblib.load("models/parkinsons.pkl")
            input_data = {
                "MDVP:Fo(Hz)": fo,
                "MDVP:Fhi(Hz)": fhi,
                "MDVP:Flo(Hz)": flo,
                "MDVP:Jitter(%)": jitter,
                "MDVP:Shimmer": shimmer,
                "RPDE": rpde,
                "DFA": dfa,
                "spread1": spread1,
                "spread2": spread2,
                "D2": d2,
                "PPE": PPE
            }
            input_scaled = prepare_input(input_data, "models/parkinsons_features.pkl", "models/parkinsons_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Parkinson's Detected", "No Parkinson's")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ---------------- Kidney Disease ----------------
elif disease == "Kidney Disease":
    st.header("💊 Chronic Kidney Disease Prediction")
    st.write("")
    st.subheader("Patient Health Information")
    st.markdown("")

    st.subheader("Vital Signs")
    age = st.slider("Age (20–90)", 20, 90, 45)
    bp = st.slider("Blood Pressure (50–180 mm/Hg)", 50, 180, 80)
    st.subheader("Urine Analysis")
    sg = st.slider("Specific Gravity (1.00–1.05)", 1.00, 1.05, 1.02)
    al = st.slider("Albumin (0–5)", 0, 5, 1)
    su = st.slider("Sugar (0–5)", 0, 5, 0)
    st.markdown("")
    st.subheader("Blood Analysis")
    bgr = st.slider("Blood Glucose Random (70–400 mg/dL)", 70, 400, 120)
    bu = st.slider("Blood Urea (10–150 mg/dL)", 10, 150, 40)
    sc = st.slider("Serum Creatinine (0.1–15.0 mg/dL)", 0.1, 15.0, 1.2)
    sod = st.slider("Sodium (100–160 mEq/L)", 100, 160, 140)
    pot = st.slider("Potassium (2.0–7.0 mEq/L)", 2.0, 7.0, 4.5)
    hemo = st.slider("Hemoglobin (3–17 g/dL)", 3.0, 17.0, 13.0)
    pcv = st.slider("Packed Cell Volume (20–60 %)", 20, 60, 40)
    wbcc = st.slider("White Blood Cell Count (2000–20000 /cmm)", 2000, 20000, 8000)
    rbcc = st.slider("Red Blood Cell Count (2–7 million/cmm)", 2.0, 7.0, 5.0)

    if st.button("Predict Kidney Disease"):
        try:
            model = joblib.load("models/kidney.pkl")
            input_data = {
                "age": age,
                "bp": bp,
                "sg": sg,
                "al": al,
                "su": su,
                "bgr": bgr,
                "bu": bu,
                "sc": sc,
                "sod": sod,
                "pot": pot,
                "hemo": hemo,
                "pcv": pcv,
                "wbcc": wbcc,
                "rbcc": rbcc
            }

            input_scaled = prepare_input(input_data, "models/kidney_features.pkl", "models/kidney_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Chronic Kidney Disease Detected", "No Kidney Disease")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

