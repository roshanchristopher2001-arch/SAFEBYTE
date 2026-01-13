
---

# 🥗 **SafeByte AI**

**Food Risk Level Prediction System using AI & Machine Learning**

---

## Project Overview

**SafeByte AI** is an AI-assisted food spoilage risk prediction system designed to support **quality assurance** in food safety. The application predicts the risk level (**Medium** or **High**) of food spoilage based on storage conditions, spoilage indicators, and selected guidelines, providing human-readable explanations for each prediction using AI. This helps streamline the rigorous quality assurance process and ensures safer food handling.


This system is needed to:

* Prevent health risks caused by spoiled food
* Reduce food wastage
* Support **Sustainable Development Goal (SDG) 3 – Good Health and Well-Being**

SafeByte AI is both:

* a **UI-based application** (built using Streamlit)
* an **ML-based system** for prediction

It also uses an **AI language model** to explain predictions in simple, understandable terms.

---

##  **Problem Statement**

Ensuring food safety and quality is a critical but often rigorous process, requiring assessment against multiple standards and guidelines. Traditional quality assurance methods are time-consuming and prone to human error, making it challenging to quickly and accurately determine the current safety status of food items. There is a need for an intelligent system that assists in monitoring food spoilage risk while streamlining quality assurance processes.

## UI Description

The application provides a **clean and user-friendly Streamlit interface**.

### User Inputs

* Dropdown to select **Food Item**
* Dropdown to select **Spoilage Signs**
* Dropdown to select **Storage Type** (Room Temperature / Fridge / Freezer)
* Numeric input for **Number of Storage Days**

### User Action

* Clicks the **“Predict Risk Level”** button

### Output Displayed

* **Predicted Risk Level**:

  * 🟥 High
  * 🟨 Medium

* **AI-generated explanation** describing:

  * Food condition
  * Storage impact
  * Spoilage likelihood

---

## Features

* Food spoilage risk level prediction
* Two risk categories: **High, Medium**
* AI-generated explanation using **LLaMA**
* Interactive and user-friendly **Streamlit UI**
* Attractive **background image–based design**

---

##  Technologies Used

* Python
* Streamlit (UI & frontend)
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Hugging Face Inference API
* Meta **LLaMA 3 – 8B Instruct**

---

##  Project Structure

```
SafeByte_AI/
│
├── app.py
├── food_spoilage_encoders.pkl
├── food_spoilage_model.pkl
├── food_spoilage_scaler.pkl
├── food_spoilage_y_encoder.pkl
├── README.md
├── requirements.txt
├── safebyte.bg.png
```

---

##  Installation & Setup

### Step 1: Python Installation

Ensure **Python 3.8 or above** is installed.

### Step 2: Clone the Project

```bash
git clone https://github.com/your-username/safebyte-ai.git
cd safebyte-ai
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### Step 4: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

##  Hugging Face API / Secrets Setup

This project uses **Hugging Face LLM** for AI explanations.

### Steps

1. Create a Hugging Face account
2. Generate an API token
3. Create a Streamlit secrets file

Create:

```
.streamlit/secrets.toml
```

Add:

```toml
HF_TOKEN = "your_token_here"
```

---

##  How to Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

##  Working / Workflow of the Application

1. User selects food-related QA parameters through the UI
2. Inputs are:

   * Encoded using label encoders
   * Scaled using a trained scaler
3. ML model predicts the **food spoilage risk level**
4. AI model generates a **human-readable explanation**
5. UI displays:

   * Risk level
   * AI explanation

---

## Sample Output

**Risk Level:** 🔴 High

**AI Explanation:**
The food shows clear spoilage signs and has been stored for an extended period. Improper storage conditions increase bacterial growth, making it unsafe for consumption.

---

##  Use Cases

* Food safety monitoring
* Household food quality checks
* Student projects in AI & ML
* Quality assurance systems

---

##  Future Enhancements

* Low Risk Prediction – Currently, no dataset is available for low-risk foods.
* CSV Integration – Allow batch prediction from CSV files.
* Ensemble Prediction – Provide a single risk level output for multiple predictions.
* Expiry-date and barcode integration

---

##  Author Details

**Roshan Christopher**

QA/QC- FMCG

---

## License / Disclaimer

This project is developed **for educational and academic purposes only**.
It should not be used as a replacement for professional food safety inspection.

---


