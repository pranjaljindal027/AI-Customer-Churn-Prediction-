# Customer Churn Dashboard with a SQLite database
# Run with:  streamlit run app.py

import os
import sqlite3
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Dashboard")
st.write("Find out why customers leave and which customers are at risk.")

folder = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(folder, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DB_PATH = os.path.join(folder, "churn.db")


# ---------------------------------------------------------------
# 1. Database functions (SQLite)
# ---------------------------------------------------------------
def create_database():
    # Convert the CSV file into a SQLite database (only the first time)
    if os.path.exists(DB_PATH):
        return
    data = pd.read_csv(CSV_PATH)
    # TotalCharges is text, so change it to numbers and drop blank rows
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()
    conn = sqlite3.connect(DB_PATH)
    data.to_sql("customers", conn, index=False)   # creates the table "customers"
    conn.close()


def load_customers():
    # Read all customers from the database (runs every time, so it is always fresh)
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    # Yes -> 1 (left), No -> 0 (stayed), empty -> not known yet
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data


def id_exists(customer_id):
    # True if a customer with this ID is already in the database
    conn = sqlite3.connect(DB_PATH)
    found = conn.execute("SELECT COUNT(*) FROM customers WHERE customerID = ?",
                         (customer_id,)).fetchone()[0]
    conn.close()
    return found > 0


def next_customer_id():
    # Automatic IDs look like NEW-0001, NEW-0002 ... (skips any that are taken)
    conn = sqlite3.connect(DB_PATH)
    number = conn.execute(
        "SELECT COUNT(*) FROM customers WHERE customerID LIKE 'NEW-%'").fetchone()[0] + 1
    conn.close()
    while id_exists("NEW-" + str(number).zfill(4)):
        number = number + 1
    return "NEW-" + str(number).zfill(4)


def add_customer(values):
    # Save one new customer in the database
    conn = sqlite3.connect(DB_PATH)
    names = ", ".join(values.keys())
    marks = ", ".join(["?"] * len(values))
    conn.execute("INSERT INTO customers (" + names + ") VALUES (" + marks + ")",
                 list(values.values()))
    conn.commit()
    conn.close()


if not os.path.exists(DB_PATH) and not os.path.exists(CSV_PATH):
    st.error("Dataset not found. Put WA_Fn-UseC_-Telco-Customer-Churn.csv "
             "in the same folder as app.py.")
    st.stop()

create_database()
df = load_customers()


# ---------------------------------------------------------------
# 2. Train the model (runs once, then it is remembered)
# ---------------------------------------------------------------
def encode(data):
    # Drop the ID and the answer, then turn text into 0/1 columns
    features = data.drop(columns=["customerID", "Churn"])
    return pd.get_dummies(features, drop_first=True, dtype=int)


@st.cache_resource
def train_model(data):
    X = encode(data)
    y = data["Churn"].astype(int)

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


# Only customers whose result is known (stayed or left) are used for training.
# New customers have no result yet, so they do not change the model.
known_customers = df.dropna(subset=["Churn"])
model, model_columns, scores, importance = train_model(known_customers)

# Churn chance (0 to 1) for EVERY customer in the database, including new ones
all_features = encode(df).reindex(columns=model_columns, fill_value=0)
risk_all = pd.Series(model.predict_proba(all_features)[:, 1], index=df.index)


def suggest_action(risk, contract, tenure):
    # Simple rules that turn the risk into an action
    if risk < 0.5:
        return "Low risk - no action needed"
    if contract == "Month-to-month":
        return "Offer a discount to move to a 1-year contract"
    if tenure < 12:
        return "Send a welcome offer and a support call"
    return "Give a loyalty reward"


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
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Customers", len(filtered))
col2.metric("Churn rate", str(round(filtered["Churn"].mean() * 100, 1)) + " %")
col3.metric("Avg monthly charges", "$" + str(round(filtered["MonthlyCharges"].mean(), 2)))
col4.metric("Avg tenure (months)", round(filtered["tenure"].mean(), 1))
col5.metric("New customers added", int(df["Churn"].isna().sum()))   # no result yet = new


# ---------------------------------------------------------------
# 5. Tabs
# ---------------------------------------------------------------
tab_charts, tab_model, tab_view, tab_add, tab_risk = st.tabs(
    ["Charts", "Model results", "View customers", "Add new customer", "Churn risk"])

# ----- Tab 1: Charts -----
with tab_charts:
    st.subheader("Churn rate by category")
    column_choice = st.selectbox(
        "Choose a column",
        ["Contract", "InternetService", "PaymentMethod", "gender",
         "SeniorCitizen", "PaperlessBilling", "Partner", "Dependents"])
    st.bar_chart(filtered.groupby(column_choice)["Churn"].mean())

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
with tab_model:
    st.write("A Random Forest model, trained on 70% of the customers and "
             "tested on the other 30%.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", round(scores["Accuracy"], 3))
    m2.metric("Precision", round(scores["Precision"], 3))
    m3.metric("Recall", round(scores["Recall"], 3))
    m4.metric("F1 score", round(scores["F1 score"], 3))

    st.subheader("Top 10 features that decide churn")
    st.bar_chart(importance.sort_values(ascending=False).head(10))

# ----- Tab 3: View existing customers -----
with tab_view:
    st.write("All customers in the database. The newest customers are at the top.")
    search = st.text_input("Search by customer ID (for example 7590 or NEW-0001)")

    show = filtered.iloc[::-1].copy()
    if search != "":
        show = show[show["customerID"].str.contains(search, case=False)]
    show["Churn"] = show["Churn"].map({1: "Yes", 0: "No"}).fillna("Not known yet")

    st.write(str(len(show)) + " customers")
    st.dataframe(show)

# ----- Tab 4: Add a new customer -----
with tab_add:
    st.write("Fill in the details and click the button. "
             "The customer is saved in the SQLite database.")
    st.caption("On Streamlit Cloud, the database file is reset when the app restarts.")

    # Message from the last time a customer was added
    if "message" in st.session_state:
        st.success(st.session_state.pop("message"))

    text_columns = ["gender", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]

    with st.form("new_customer_form"):
        typed_id = st.text_input(
            "Customer ID (leave empty to get an automatic ID like NEW-0001)")

        boxes = st.columns(3)
        tenure = boxes[0].slider("Tenure (months)", 0, 72, 12)
        monthly = boxes[1].slider("Monthly charges ($)", 18, 120, 70)
        senior = boxes[2].selectbox("Senior citizen", ["No", "Yes"])

        # The options come from the database, so the spelling always matches
        choices = {}
        for i, name in enumerate(text_columns):
            choices[name] = boxes[i % 3].selectbox(name, sorted(df[name].unique()))

        submitted = st.form_submit_button("Add customer")

    if submitted:
        typed_id = typed_id.strip()              # remove spaces at the start and end
        if typed_id == "":
            new_id = next_customer_id()          # no ID typed, so make one
        elif id_exists(typed_id):
            new_id = None                        # this ID is already used
            st.error("Customer ID " + typed_id + " already exists. Use a different ID.")
        else:
            new_id = typed_id

    if submitted and new_id is not None:
        values = {
            "customerID": new_id,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(tenure * monthly),
            "Churn": None,                     # we do not know yet if they will leave
        }
        values.update(choices)
        add_customer(values)

        # Save a message, then reload the page so every tab shows the new customer
        st.session_state["message"] = "Customer " + new_id + " was added to the database."
        st.rerun()

# ----- Tab 5: Churn risk -----
with tab_risk:
    st.write("The model gives every customer a chance of leaving. "
             "New customers appear here as soon as they are added.")
    st.caption("Customers already in the dataset were used to train the model, "
               "so their scores look more certain than the scores of new customers.")

    # Part 1: look at one customer
    ids = list(df["customerID"])[::-1]          # newest first
    chosen_id = st.selectbox("Choose a customer", ids)
    position = df.index[df["customerID"] == chosen_id][0]
    row = df.loc[position]
    risk = float(risk_all.loc[position])

    st.subheader("Churn risk of " + chosen_id + ": " + str(round(risk * 100, 1)) + " %")
    st.progress(risk)

    if row["Churn"] == 1:
        st.info("This customer has already left.")
    elif risk < 0.5:
        st.success(suggest_action(risk, row["Contract"], row["tenure"]))
    else:
        st.error(suggest_action(risk, row["Contract"], row["tenure"]))

    st.write("Customer details")
    st.dataframe(row.astype(str).to_frame("value"))

    # Part 2: the 10 current customers most likely to leave
    st.subheader("Top 10 current customers most at risk")
    risk_table = df[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn"]].copy()
    risk_table["Risk %"] = (risk_all * 100).round(1)

    current = risk_table[risk_table["Churn"] != 1]     # stayed or new
    top10 = current.sort_values("Risk %", ascending=False).head(10).copy()
    top10["Suggested action"] = [
        suggest_action(r / 100, c, t)
        for r, c, t in zip(top10["Risk %"], top10["Contract"], top10["tenure"])
    ]
    st.dataframe(top10.drop(columns=["Churn"]))
