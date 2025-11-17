# AIML_miniproject
BudgetWise: An Intelligent Personal Finance Management System 

BudgetWise is an AI-driven personal finance tool that helps users track expenses, scan bills using OCR, get savings insights, forecast future spending, and receive personalized budgeting recommendations using machine learning and Gemini Flash API.

Features

* *Bill OCR Scanner* – Extracts text & amounts from uploaded receipts
* *Expense Prediction* – XGBoost model with feature engineering
* *Time-Series Forecasting* – Prophet model for monthly expense trends
* *AI Recommendations* – Personalized finance advice via Gemini Flash
* *Savings Goal Tracker* – Set goals, add contributions, monitor progress
* *User Dashboard* – Clean UI for transactions, summaries & charts
* *Automated Reports* – Monthly summaries generated from user data

## 📁 *Project Structure*


AIML_MINIPROJECT/
│ app.py
│ bill_ocr_processor.py
│ database.py
│ models.py
│ preprocess.py
│ recommendation_engine.py
│ report_generator.py
│ requirements.txt
│ budgetwise_finance_dataset.csv
│ cleaned_data.csv
│ feature_names.pkl
│ feature_scaler.pkl
│ label_encoders.pkl
│ improved_xgb_model.pkl
│ improved_prophet_model.pkl
│ improved_user_profiles.pkl
│ xgb_model.pkl
│ image.png
│
├── templates/
│   ├── dashboard.html
│   ├── home.html
│   ├── login.html
│   ├── signup.html
│
├── static/
│
└── uploads/


---

## ⚙ *Installation & Setup*

### 1️⃣ Clone the repository

bash
git clone <your-repo-url>
cd AIML_MINIPROJECT


### 2️⃣ Create virtual environment

bash
python -m venv venv
venv\Scripts\activate   # Windows


### 3️⃣ Install dependencies

bash
pip install -r requirements.txt


### 4️⃣ Add your *Gemini API Key*

Create a .env file:


GEMINI_API_KEY=your_key_here


### 5️⃣ Run the application

bash
python app.py


Application will run at:
👉 *[http://localhost:5000](http://localhost:5000)*


## 🤖 *ML Models Used*

* *XGBoost* for expense prediction

  * R² (Test): ~0.97
  * MAE: ~₹3,900
  * RMSE: ~₹19,000
* *Prophet* for time-series forecasting
* *User Profiles* generated with 150 synthetic users
* *OCR* using PaddleOCR or Tesseract

Models saved as:

improved_xgb_model.pkl  
improved_prophet_model.pkl  
improved_user_profiles.pkl  

## 🧠 *Gemini AI Recommendations*

BudgetWise uses Google Gemini Flash to generate:

* budgeting tips
* spending optimization
* savings plan guidance
* risk alerts
* category-wise suggestions

Prompt example:

User Profile:
- Monthly Income: ₹{income}
- Total Expenses: ₹{expenses}
- Savings Rate: {savings_rate}%
- Top Categories: {top_categories}

Give 3–5 personalized budgeting recommendations.


## 🖥 *User Interface Pages*

* *Home Page*
* *Login / Signup*
* *Dashboard*
* *OCR Receipt Upload*

## 👩‍💻 *Team*

Srushti Tarate
Feyoni Shah

---

## 📜 *License*

This project is for educational purposes.

