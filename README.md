# AI-Based Customer Churn Prediction and Retention System

A machine learning project and web dashboard that shows why telecom customers leave, predicts which customers are likely to leave next, and suggests how to keep them.

# Live dashboard 

https://pu5hbpfu2emjdr8ev8ydnp.streamlit.app/ 

# Problem statement

Customer churn means a customer stops using a company's service. Winning a new customer costs more than keeping an existing one, so telecom companies want to know early which customers are at risk. This project uses past customer data to learn the pattern of who leaves, then scores customers by risk and suggests a retention action.

# Dataset

- Name: Telco Customer Churn (IBM sample dataset, from Kaggle)
- File: WA_Fn-UseC_-Telco-Customer-Churn.csv
- About 7,000 customers, one row per customer
- Columns include contract type, internet service, payment method, tenure (months with the company), monthly charges, total charges and Churn (Yes or No)

# How it works

1. Load the data: read the CSV file into a table using Pandas.
2. Clean the data: fix blank values in TotalCharges, remove the few bad rows, and change Churn from Yes/No to 1/0.
3. Explore: charts show how churn changes with contract type, internet service, payment method and more.
4. Prepare: drop customerID (it does not help prediction) and convert text columns into numbers.
5. Split: 70% of customers are used for training and 30% are kept hidden for testing.
6. Train: a Random Forest model learns from the training data. A Random Forest is many small decision trees that vote on the answer.
7. Evaluate: the model predicts the hidden 30%, and its answers are compared with the real ones.
8. Retention: each customer gets a churn risk score, and simple rules turn it into a suggested action.

# Dashboard features

- Key numbers at the top: customers, churn rate, average charges and average tenure
- Sidebar filter to view data by contract type
- Charts tab: churn by contract, internet service, payment method and more
- Model results tab: accuracy, precision, recall, F1 score and confusion matrix
- Predict a customer tab: enter a customer's details to get a churn risk and a suggested action
- Data tab: view the filtered table

# Model evaluation

- Accuracy: how many predictions were right overall
- Precision: of the customers predicted to leave, how many really left
- Recall: of the customers who really left, how many the model found
- F1 score: a balance of precision and recall
- Confusion matrix: a table of correct and wrong predictions

# Retention actions

The risk score is turned into an action with simple rules:

- Low risk: no action needed
- Month-to-month contract: offer a discount to move to a longer contract
- New customer: send a welcome offer and a support call
- Other at-risk customers: give a loyalty reward

# Project files

- app.py : the Streamlit dashboard code
- requirements.txt : the Python libraries the app needs
- WA_Fn-UseC_-Telco-Customer-Churn.csv : the dataset
- README.md : this file

# How to run on your computer

1. Install Python from python.org.
2. Put all the project files in one folder.
3. Open a terminal in that folder and install the libraries:

```
pip install -r requirements.txt
```

4. Start the dashboard:

```
streamlit run app.py
```

5. Your browser opens the dashboard at http://localhost:8501.

# Technologies used

- Python
- Pandas for data handling
- Scikit-learn for the Random Forest model
- Streamlit for the web dashboard
- GitHub and Streamlit Community Cloud for hosting

# Future improvements

- Compare more models, such as Logistic Regression and XGBoost
- Add more filters and charts to the dashboard
- Connect to live customer data instead of a CSV file
