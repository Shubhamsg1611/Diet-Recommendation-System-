import streamlit as st
import pandas as pd
import joblib

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Personalized Diet Recommendation System",
    layout="wide"
)

# ------------------- LOAD MODEL -------------------
model = joblib.load("xgb_diet_model.pkl")
label_encoder = joblib.load("diet_label_encoder.pkl")
model_columns = joblib.load("diet_model_columns.pkl")

# ------------------- CSS -------------------
st.markdown("""
<style>
.stApp {
    background-image: url(https://media.istockphoto.com/id/1078560062/photo/balanced-diet-food-background.jpg);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Card wrapper */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.95);
    padding: 3rem;
    border-radius: 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.30);
    max-width: 1200px;
    margin: 3rem auto;
}

/* Title */
.app-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1f7a4d;
    margin-bottom: 0.3rem;
}

/* Subtitle */
.app-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #555;
    margin-bottom: 2.2rem;
}

/* Footer */
.app-footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #ddd;
    font-size: 0.95rem;
    color: #555;
}
.app-footer strong {
    color: #1f7a4d;
}
</style>
""", unsafe_allow_html=True)

# ------------------- CARD CONTAINER -------------------
with st.container():

    st.markdown("<div class='app-title'>Personalized Diet Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Enter patient details to receive a personalized diet plan</div>", unsafe_allow_html=True)

    # ------------------- INPUTS -------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal & Body Metrics")
        age = st.slider("Age (years)", 18, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", 120, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        daily_calories = st.slider("Daily Caloric Intake (kcal)", 1200, 4500, 2200)
        weekly_exercise_hours = st.slider("Weekly Exercise Hours", 0, 20, 4)

    with col2:
        st.subheader("Health & Lifestyle")
        physical_activity = st.selectbox(
            "Physical Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
        )
        disease_type = st.selectbox(
            "Primary Disease",
            ["None", "Diabetes", "Hypertension", "Cardiac", "Obesity"]
        )
        severity = st.selectbox("Disease Severity", ["Mild", "Moderate", "Severe"])
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 350, 190)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        glucose = st.number_input("Glucose (mg/dL)", 70, 300, 100)

    # ------------------- DIETARY PREFERENCES -------------------
    st.subheader("Dietary Preferences")

    col3, col4 = st.columns(2)
    with col3:
        dietary_restriction = st.selectbox(
            "Dietary Restriction",
            ["None", "Vegetarian", "Vegan", "Gluten-Free", "Lactose-Free"]
        )

    with col4:
        allergy = st.selectbox(
            "Allergy",
            ["None", "Nuts", "Dairy", "Seafood", "Gluten"]
        )

    preferred_cuisine = st.selectbox(
        "Preferred Cuisine",
        ["Indian", "Mediterranean", "Continental", "Asian"]
    )

    adherence = st.selectbox(
        "Adherence to Diet Plan",
        ["Low", "Medium", "High"]
    )

    # ------------------- CALCULATIONS -------------------
    bmi = round(weight / (height / 100) ** 2, 2)
    bmr = round(10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161), 2)

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }[physical_activity]

    tdee = round(bmr * activity_factor, 2)
    calorie_balance = round(daily_calories - tdee, 2)

    st.info(
        f"**BMI:** {bmi} | **BMR:** {bmr} | **TDEE:** {tdee} | **Calorie Balance:** {calorie_balance}"
    )

    # ------------------- MODEL INPUT -------------------
    numeric = {
        "Age": age,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi,
        "BMR": bmr,
        "TDEE": tdee,
        "Calorie_Balance": calorie_balance,
        "Daily_Caloric_Intake": daily_calories,
        "Cholesterol_mg/dL": cholesterol,
        "Blood_Pressure_mmHg": blood_pressure,
        "Glucose_mg/dL": glucose,
        "Weekly_Exercise_Hours": weekly_exercise_hours,
    }

    categorical = {
        f"Gender_{gender}": 1,
        f"Physical_Activity_Level_{physical_activity}": 1,
        f"Disease_Type_{disease_type}": 1,
        f"Severity_{severity}": 1,
        f"Dietary_Restrictions_{dietary_restriction}": 1,
        f"Preferred_Cuisine_{preferred_cuisine}": 1,
        f"Allergies_{allergy}": 1,
        f"Adherence_to_Diet_Plan_{adherence}": 1,
    }

    final_input = dict.fromkeys(model_columns, 0)
    final_input.update(numeric)
    final_input.update(categorical)

    input_df = pd.DataFrame([final_input])[model_columns]

    # ------------------- PREDICTION -------------------
    if st.button("Get Diet Recommendation"):
        prediction = model.predict(input_df)
        diet = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Recommended Diet Plan: **{diet}**")

    # ------------------- FOOTER -------------------
    st.markdown("""
    <div class="app-footer">
        Developed by <strong>Shubham Ghanwat</strong><br>
        Data Science & Machine Learning Practitioner
    </div>
    """, unsafe_allow_html=True)
