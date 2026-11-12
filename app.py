# Customer Churn Dashboard (simple version)
# Run with:  streamlit run app.py

import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Dashboard")
st.write("Find out why customers leave and which customers are at risk.")


# ---------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    folder = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(folder, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        return None
    data = pd.read_csv(csv_path)
    # TotalCharges is text, so change it to numbers and drop blank rows
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()
    # Yes -> 1 (left), No -> 0 (stayed)
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data


df = load_data()

if df is None:
    st.error("Dataset not found. Put WA_Fn-UseC_-Telco-Customer-Churn.csv "
             "in the same folder as app.py.")
    st.stop()


# ---------------------------------------------------------------
# 2. Train the model (runs once, then it is remembered)
# ---------------------------------------------------------------
@st.cache_resource
def train_model(data):
    model_data = data.drop(columns=["customerID"])
    model_data = pd.get_dummies(model_data, drop_first=True, dtype=int)

    X = model_data.drop(columns=["Churn"])
    y = model_data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=17, stratify=y)

    model = RandomForestClassifier(random_state=46)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 score": f1_score(y_test, y_pred),
    }
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return model, list(X.columns), scores, importance


model, model_columns, scores, importance = train_model(df)


# ---------------------------------------------------------------
# 3. Sidebar filter
# ---------------------------------------------------------------
st.sidebar.header("Filter customers")
contract_list = sorted(df["Contract"].unique())
chosen_contracts = st.sidebar.multiselect(
    "Contract type", contract_list, default=contract_list)

filtered = df[df["Contract"].isin(chosen_contracts)]

if filtered.empty:
    st.warning("No customers match the filter. Choose at least one contract type.")
    st.stop()


# ---------------------------------------------------------------
# 4. Key numbers at the top
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", len(filtered))
col2.metric("Churn rate", str(round(filtered["Churn"].mean() * 100, 1)) + " %")
col3.metric("Avg monthly charges", "$" + str(round(filtered["MonthlyCharges"].mean(), 2)))
col4.metric("Avg tenure (months)", round(filtered["tenure"].mean(), 1))


# ---------------------------------------------------------------
# 5. Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Model results", "Predict a customer", "Data"])

# ----- Tab 1: Charts -----
with tab1:
    st.subheader("Churn rate by category")
    column_choice = st.selectbox(
        "Choose a column",
        ["Contract", "InternetService", "PaymentMethod", "gender",
         "SeniorCitizen", "PaperlessBilling", "Partner", "Dependents"])
    churn_by_column = filtered.groupby(column_choice)["Churn"].mean()
    st.bar_chart(churn_by_column)

    left, right = st.columns(2)
    with left:
        st.subheader("Average monthly charges")
        st.write("0 = stayed, 1 = left")
        st.bar_chart(filtered.groupby("Churn")["MonthlyCharges"].mean())
    with right:
        st.subheader("Average tenure (months)")
        st.write("0 = stayed, 1 = left")
        st.bar_chart(filtered.groupby("Churn")["tenure"].mean())

# ----- Tab 2: Model results -----
with tab2:
    st.write("A Random Forest model, trained on 70% of the customers and "
             "tested on the other 30%.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", round(scores["Accuracy"], 3))
    m2.metric("Precision", round(scores["Precision"], 3))
    m3.metric("Recall", round(scores["Recall"], 3))
    m4.metric("F1 score", round(scores["F1 score"], 3))

    st.subheader("Top 10 features that decide churn")
    st.bar_chart(importance.sort_values(ascending=False).head(10))

# ----- Tab 3: Predict one customer -----
with tab3:
    st.write("Enter the details of a customer to see their chance of leaving.")

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c1.slider("Monthly charges ($)", 18, 120, 70)
    contract = c2.selectbox("Contract", sorted(df["Contract"].unique()))
    internet = c2.selectbox("Internet service", sorted(df["InternetService"].unique()))
    payment = c3.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
    paperless = c3.selectbox("Paperless billing", sorted(df["PaperlessBilling"].unique()))

    # Start from the most common value in every column, then change
    # the columns that the user entered.
    features = df.drop(columns=["customerID", "Churn"])
    new_customer = features.mode().iloc[[0]].copy()
    new_customer["tenure"] = tenure
    new_customer["MonthlyCharges"] = monthly
    new_customer["TotalCharges"] = tenure * monthly
    new_customer["Contract"] = contract
    new_customer["InternetService"] = internet
    new_customer["PaymentMethod"] = payment
    new_customer["PaperlessBilling"] = paperless

    # Turn text into 0/1 columns in the same way as the training data
    combined = pd.concat([features, new_customer])
    combined = pd.get_dummies(combined, drop_first=True, dtype=int)
    new_row = combined.tail(1)[model_columns]

    risk = model.predict_proba(new_row)[0, 1]
    st.subheader("Churn risk: " + str(round(risk * 100, 1)) + " %")
    st.progress(float(risk))

    # Simple rules that turn the risk into an action
    if risk < 0.5:
        st.success("Low risk - no action needed.")
    elif contract == "Month-to-month":
        st.error("High risk - offer a discount to move to a 1-year contract.")
    elif tenure < 12:
        st.error("High risk - send a welcome offer and a support call.")
    else:
        st.error("High risk - give a loyalty reward.")

# ----- Tab 4: Data -----
with tab4:
    st.write(str(len(filtered)) + " customers")
    st.dataframe(filtered)