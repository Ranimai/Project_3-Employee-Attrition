import streamlit as st
import pandas as pd
import pickle

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("lr_scaled_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Set the page config
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📁Employee Attrition Analysis")
    menu = st.radio("⚙️ Navigate", ["Home", "Predict Employee Attrition"])

# Home Page
if menu == "Home":
    st.markdown("<h1 style='text-align: center; color: black;'>📊 Dashboard Home</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>👨‍💼Employee Insights Dashboard</h2>", unsafe_allow_html=True)

    df = pd.read_csv("cleaned_data_with_target.csv")

    st.write(df.head())

 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ⚠️ High-Risk Employees")
        high_risk = df[df['Attrition'] > 0.7][['remainder_age', 'remainder_totalworkingyears', 'Attrition']].head(10)
        st.dataframe(high_risk, use_container_width=True)

    with col2:
        st.markdown("#### 😀 High Job Satisfaction")
        high_satisfaction = df[df['remainder_jobsatisfaction'] >= 4][['remainder_joblevel', 'remainder_jobsatisfaction', 'Attrition']].head(10)
        st.dataframe(high_satisfaction, use_container_width=True)

    with col3:
        st.markdown("#### ♻️ Work Life Balance ")
        life_balance = df[df['remainder_monthlyincome'] > 80][['ordinal_jobrole', 'remainder_monthlyincome', 'Attrition']].head(10)
        st.dataframe(life_balance, use_container_width=True)



# Prediction Page
elif menu == "Predict Employee Attrition":
    st.markdown("<h1 style='text-align: center; color: brown;'>🔮 Attrition Prediction Page View</h1>", unsafe_allow_html=True)
    st.subheader("🔍Predict Employee Attrition")

    with st.form("attrition_form"):
        business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
        job_role = st.selectbox("Job Role", ['Human Resources', 'Laboratory Technician', 'Healthcare Representative',
                                         'Sales Representative', 'Sales Executive', 'Manager', 'Manufacturing Director',
                                         'Research Scientist', 'Research Director'])
        education_field = st.selectbox("Education Field", ['Human Resources', 'Other', 'Technical Degree',
                                                       'Marketing', 'Medical', 'Life Sciences'])
        over_time = st.selectbox("Over Time", ["Yes", "No"])
        department = st.selectbox("Department", ["Research & Development", "Sales", "HR"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
        age = st.number_input("Age", min_value=18, max_value=60, value=25)
        distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=30, value=5)
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=40, value=2)
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=40, value=2)

        submitted = st.form_submit_button("📄 Predict Attrition")

    if submitted:
        input_data = pd.DataFrame({
            "BusinessTravel": [business_travel],
            "JobRole": [job_role],
            "EducationField": [education_field],
            "OverTime": [over_time],
            "Department": [department],
            "Gender": [gender],
            "MaritalStatus": [marital_status],
            "Age": [age],
            "DistanceFromHome": [distance_from_home],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "JobInvolvement": [job_involvement],
            "JobLevel": [job_level],
            "JobSatisfaction": [job_satisfaction],
            "MonthlyIncome": [monthly_income],
            "StockOptionLevel": [stock_option_level],
            "TotalWorkingYears": [total_working_years],
            "YearsAtCompany": [years_at_company],
            "YearsInCurrentRole": [years_in_current_role],
            "YearsWithCurrManager": [years_with_curr_manager]
        })


        x_transformed = preprocessor.transform(input_data)

        scaled_input = scaler.transform(x_transformed)

        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # Display results
        st.markdown("### 🔍 Prediction Result:")
        if prediction == 1:
            st.error(f"😟👨‍💼 Employee likely to **leave** the company (Probability: {probability:.2f})")
        else:
            st.success(f"😊👨‍💼 Employee likely to **stay** in the company (Probability: {probability:.2f})")

            







