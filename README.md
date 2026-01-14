# Diet Recommendation System using Machine Learning

An end-to-end **Machine Learning Diet Recommendation System** that predicts a personalized diet plan based on user health metrics.  
The project uses an **XGBoost Classifier** and is deployed as an interactive **Streamlit web application**.

---

## Live Application

🔗 **Streamlit Deployment**  
https://diet-recommendations-appsg.streamlit.app/

---

## Introduction

Maintaining a healthy diet is essential for preventing lifestyle diseases such as diabetes, obesity, and hypertension.  
This project aims to provide **data-driven diet recommendations** by analyzing key health indicators using Machine Learning.

The system:
- Accepts health inputs from users
- Processes them using a trained ML model
- Predicts an appropriate diet category instantly

---

## Features

- End-to-end ML pipeline (data → model → deployment)
- Trained **XGBoost classification model**
- Clean and interactive **Streamlit UI**
- Fast real-time predictions
- Reusable ML artifacts (`.pkl` files)
- Easy local and cloud deployment

---

## Tech Stack

### Programming
- Python 3.x

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Model Persistence
- Joblib

### Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit

---

## Machine Learning Pipeline

1. **Data Collection**
   - Health and nutrition dataset in CSV format

2. **Data Preprocessing**
   - Handling missing values
   - Feature transformation
   - Encoding categorical variables

3. **Feature Engineering**
   - Selection of relevant health indicators
   - Column alignment for inference

4. **Model Training**
   - XGBoost Classifier
   - Hyperparameter tuning
   - Model evaluation

5. **Model Serialization**
   - Saving trained model
   - Saving encoders and column metadata

6. **Deployment**
   - Streamlit web interface
   - Real-time prediction pipeline

---

## Input Parameters

The application takes the following inputs (example):

- Age  
- Gender  
- BMI  
- Glucose Level  
- Blood Pressure  
- Other health-related indicators  

*(Exact features depend on the trained dataset and model columns.)*

---

## Output

- **Predicted Diet Recommendation**
- Displayed instantly on the web interface
- Designed for clarity and usability

---

## Use Cases

- Healthcare analytics projects

- Diet and nutrition recommendation systems

- Machine Learning portfolio projects

- Streamlit deployment demonstrations

- Academic and practical ML applications

---

## Author

Shubham Ghanwat


GitHub: https://github.com/Shubhamsg1611


