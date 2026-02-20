
<h1 align="center"><b>AI Disease Prediction System</b></h1>

<p align="center">
An AI-powered web application that predicts the likelihood of multiple diseases such as <b>Diabetes, Heart Disease, Parkinson's and Chronic Kidney Disease</b> based on user-provided health parameters.
<br>
Built using <b>Python</b> and <b>Streamlit</b>, this system leverages machine learning models to assist in early disease detection.
</p>

---

## 📌 Project Overview

This project provides an interactive web-based platform where users can input medical parameters and receive real-time disease risk predictions.

It integrates:
- Machine Learning model training
- Data preprocessing pipelines
- Model serialization
- Streamlit-based deployment

---

## 🧪 Features

- ✅ Multi-Disease Prediction (Diabetes, Heart, Parkinson's, Kidney)
- ✅ Interactive Streamlit UI
- ✅ Trained ML models with saved scalers & feature sets
- ✅ Confidence score display
- ✅ Feature importance visualization
- ✅ Modular and scalable architecture
- ✅ Easy local deployment

---

## 💻 Tech Stack

**Programming Language:** Python  
**Machine Learning:** Scikit-learn, XGBoost  
**Web Framework:** Streamlit  
**Data Handling:** Pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Model Serialization:** joblib / pickle  
**Deployment:** Streamlit Cloud  

---

## 📁 Folder Structure

```bash
AI_Disease_Predictor/
│
├── app/
│   └── streamlit_app.py            # Main Streamlit application
│
├── models/                         # Trained ML models and preprocessing files
│   ├── diabetes.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   ├── parkinsons.pkl
│   └── ...
│
├── notebooks/                      # Model training notebooks
│   ├── diabetes_model.ipynb
│   ├── heart_model.ipynb
│   ├── kidney_model.ipynb
│   └── parkinsons_model.ipynb
│
├── data/                           # Dataset files
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney_disease.csv
│   └── parkinsons.csv
│
├── requirements.txt
└── README.md
```
## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dataqubit404/AI_Disease_Predictor.git
cd AI_Disease_Predictor
```

### 2️⃣ Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app/streamlit_app.py
```

Open your browser and go to:

`http://localhost:8501`

## 📊 Model Training

The machine learning models are trained using the datasets located in the `data/` folder. Jupyter notebooks in the `notebooks/` directory provide step-by-step guidance on training models for each disease:

- **Diabetes:** notebooks/diabetes_model.ipynb  
- **Heart Disease:** notebooks/heart_model.ipynb  
- **Chronic Kidney Disease:** notebooks/kidney_model.ipynb  
- **Parkinson's Disease:** notebooks/parkinsons_model.ipynb  

Each notebook includes:

- Data preprocessing  
- Feature selection  
- Model training  
- Model evaluation  
- Saving trained models and scalers  

After training, models are stored inside the `models/` directory for deployment.

---

## 🌐 Deployment

The application is deployed using **Streamlit**.

### 🚀 Run Locally

```bash
streamlit run app/streamlit_app.py
```
## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

You are free to:

- Use  
- Modify  
- Distribute  
- Publish  
- Sublicense  

With proper attribution.

---

## 📬 Contact

For any inquiries, collaborations, or contributions, please contact:

👨‍💻 **GitHub:** https://github.com/dataqubit404  

You can also open an issue in the repository for questions or suggestions.



