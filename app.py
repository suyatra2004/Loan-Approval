import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title("💰 Loan Approval Prediction App")
st.caption("Developed by Mandril Sircar")
st.write("Enter your details below to check if your loan will be approved.")

df = pd.read_csv("loan_data.csv")
X = df.drop("Approved", axis=1)
y = df["Approved"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

st.write("Fill in this form to know the details.")

with st.form("loan_form"):
    income = st.number_input("Income", min_value=0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=0, step=1)
    debt_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, step=0.01)
    employment_years = st.number_input("Employment Years", min_value=0, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    sample = [[income, credit_score, debt_ratio, employment_years]]
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][prediction]
    if prediction == 1:
        st.success(f"✅ Approved (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Not Approved (Confidence: {probability:.2%})")
