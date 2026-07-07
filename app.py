
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

left, right = st.columns(2)

with left:
    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen",[0,1])
    partner = st.selectbox("Partner",["Yes","No"])
    dependents = st.selectbox("Dependents",["Yes","No"])
    tenure = st.number_input("Tenure (Months)",0,72,12)
    phone = st.selectbox("Phone Service",["Yes","No"])
    multiple = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
    security = st.selectbox("Online Security",["Yes","No","No internet service"])
    backup = st.selectbox("Online Backup",["Yes","No","No internet service"])

with right:
    protection = st.selectbox("Device Protection",["Yes","No","No internet service"])
    support = st.selectbox("Tech Support",["Yes","No","No internet service"])
    tv = st.selectbox("Streaming TV",["Yes","No","No internet service"])
    movies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])
    contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing",["Yes","No"])
    payment = st.selectbox("Payment Method",[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])
    monthly = st.number_input("Monthly Charges",0.0,1000.0,70.0)
    total = st.number_input("Total Charges",0.0,100000.0,850.0)

if st.button("🔍 Predict Customer Churn", use_container_width=True):
    d = {c:0 for c in feature_columns}
    d["SeniorCitizen"]=senior
    d["tenure"]=tenure
    d["MonthlyCharges"]=monthly
    d["TotalCharges"]=total

    if gender=="Male": d["gender_Male"]=1
    if partner=="Yes": d["Partner_Yes"]=1
    if dependents=="Yes": d["Dependents_Yes"]=1
    if phone=="Yes": d["PhoneService_Yes"]=1

    if multiple=="Yes":
        d["MultipleLines_Yes"]=1
    elif multiple=="No phone service":
        d["MultipleLines_No phone service"]=1

    if internet=="Fiber optic":
        d["InternetService_Fiber optic"]=1
    elif internet=="No":
        d["InternetService_No"]=1

    for prefix,val in [
        ("OnlineSecurity",security),
        ("OnlineBackup",backup),
        ("DeviceProtection",protection),
        ("TechSupport",support),
        ("StreamingTV",tv),
        ("StreamingMovies",movies),
    ]:
        if val=="Yes":
            d[f"{prefix}_Yes"]=1
        elif val=="No internet service":
            d[f"{prefix}_No internet service"]=1

    if contract=="One year":
        d["Contract_One year"]=1
    elif contract=="Two year":
        d["Contract_Two year"]=1

    if paperless=="Yes":
        d["PaperlessBilling_Yes"]=1

    if payment=="Credit card (automatic)":
        d["PaymentMethod_Credit card (automatic)"]=1
    elif payment=="Electronic check":
        d["PaymentMethod_Electronic check"]=1
    elif payment=="Mailed check":
        d["PaymentMethod_Mailed check"]=1

    input_df = pd.DataFrame([d])[feature_columns]
    input_df[["tenure","MonthlyCharges","TotalCharges"]] = scaler.transform(
        input_df[["tenure","MonthlyCharges","TotalCharges"]]
    )

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if pred==1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.metric("Churn Probability", f"{prob:.2%}")
    st.progress(float(prob))

st.markdown("---")
st.caption("Built with Streamlit • Scikit-learn • Random Forest")
