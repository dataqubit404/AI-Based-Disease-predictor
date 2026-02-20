<h1 align="center"><b>AI Disease Prediction System</b></h1>

 An AI-powered web application that predicts the likelihood of diseases such as Diabetes, Heart Disease, Parkinson's and Chronic Kidney Disease based on user-provided health parameters. <br>Built using Python and Streamlit, this system leverages machine learning models to assist in early disease detection.
<br>

## **📌Project Overview**

This project aims to provide users with an interactive platform to assess their risk for certain diseases. By inputting specific health metrics, users receive predictions that can guide them toward seeking medical advice or making lifestyle changes.
<br>

## **🧪Features**

Disease Prediction: Predicts the likelihood of Diabetes, Heart Disease, Parkinson's and Chronic Kidney Disease based on user inputs.

Interactive Interface: User-friendly forms to enter health parameters.

Model Insights: Displays prediction results along with model confidence scores.

Visualization: Graphical representation of feature importance and prediction probabilities.

## **💻Tech Stack**

Programming Language: Python

Machine Learning Libraries: Scikit-learn, XGBoost

Web Framework: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Serialization: joblib

Deployment: Streamlit Cloud
<br>

## **📁 Folder Structure**
```bash
AI_Disease_Predictor/
├── app/
│   └── streamlit_app.py            # Main Streamlit application
│
├── models/                         # Trained ML models and preprocessing files
│   ├── diabetes.pkl                # Diabetes prediction model
│   ├── diabetes_features.pkl       # Feature list for Diabetes model
│   ├── diabetes_scaler.pkl         # Scaler for Diabetes model
│
│   ├── heart.pkl                   # Heart Disease prediction model
│   ├── heart_features.pkl          # Feature list for Heart model
│   ├── heart_scaler.pkl            # Scaler for Heart model
│
│   ├── kidney.pkl                  # Chronic Kidney Disease prediction model
│   ├── kidney_features.pkl         # Feature list for Kidney model
│   ├── kidney_scaler.pkl           # Scaler for Kidney model
│
│   ├── parkinsons.pkl              # Parkinson's Disease prediction model
│   ├── parkinsons_features.pkl     # Feature list for Parkinson's model
│   └── parkinsons_scaler.pkl       # Scaler for Parkinson's model
│
├── notebooks/
│   ├── diabetes_model.ipynb        # Diabetes model training notebook
│   ├── heart_model.ipynb           # Heart Disease model training notebook
│   ├── kidney_model.ipynb          # Chronic Kidney Disease model training notebook
│   └── parkinsons_model.ipynb      # Parkinson's model training notebook
│
├── data/
│   ├── diabetes.csv                # Dataset for Diabetes
│   ├── heart.csv                   # Dataset for Heart Disease
│   ├── kidney_disease.csv          # Dataset for Chronic Kidney Disease
│   └── parkinsons.csv              # Dataset for Parkinson's
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

```

## 🚀 Getting Started
1.&nbsp;Clone the Repository
```bash
git clone https://github.com/Vaenvoice/AI_Disease_Predictor.git
```
```bash
cd AI_Disease_Predictor
```
2.&nbsp;Set Up a Virtual Environment

For Windows:

```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the Application Locally
```bash
streamlit run app/streamlit_app.py
```

#### Open your browser and navigate to http://localhost:8501 to interact with the application.

## 📊 Model Training

The machine learning models are trained using the datasets located in the data/ folder. Jupyter notebooks in the notebooks/ directory provide step-by-step guidance on training models for each disease:

Diabetes: notebooks/diabetes_model.ipynb

Heart Disease: notebooks/heart_model.ipynb

Parkinson's: notebooks/parkinsons_model.ipynb

## 🌐 Deployment

The application is deployed on [Streamlit](https://streamlit.io/)
. Once deployed, you can access the live application via the provided localhost URL.

## 📄 License

This project is licensed under the MIT License - see the LICENSE
 file for details.

## 📬 Contact

For any inquiries or contributions, please contact <br>[Vaenvoice](https://github.com/Vaenvoice)
<br>[proxybinder](https://github.com/proxybinder)

