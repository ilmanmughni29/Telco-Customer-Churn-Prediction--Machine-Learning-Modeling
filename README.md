<h1> Telco Customer Churn Prediction - Machine Learning Modeling </h1>

## 1. Project Overview
**Context:** <br>
Telecommunication companies face significant challenges in retaining their customers. Customer churn (customers who stop using their services) poses a serious threat to company revenue. Acquiring new customers is significantly more expensive than retaining existing ones — industry research shows that new customer acquisition costs can be 5-25x higher.

This dataset represents the customer profile of a telecommunications company, including the services they use, subscription length, contract type, and monthly billing.

**Target:** <br>
- `No`: Customers have not churned (still active)
- `Yes`: Customers have churned (unsubscribed)

**Problem Statement:** <br>
Companies struggle to identify which customers are likely to churn before they actually leave. Without this predictive capability, retention teams cannot proactively intervene with the right customers.

**Goals:** <br>
To build a machine learning model capable of predicting the likelihood of a customer churning, so the company can:
1. Implement proactive and targeted retention interventions
2. Allocate retention program budgets more efficiently
3. Understand the key factors driving churn

**Analytic Approach:** <br>
We will analyze data patterns of churned and non-churned customers, then build a binary classification model to predict the probability of churn for each customer.

**Metric Evaluation:**

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | True Negative (TN) | False Positive (FP) |
| **Actual: Churn** | False Negative (FN) | True Positive (TP) |

- **False Positive (FP)**: A customer is predicted to churn when they actually do not -- the company incurs unnecessary retention costs.
- **False Negative (FN)**: A customer is predicted not to churn when they actually do -- the company loses the customer without having the opportunity to retain them.

The consequences of **False Negative** are greater because losing a customer means losing long-term revenue. However, we also want to avoid too many False Positives to prevent inflated retention costs.

**The main metrics used:** <br>
**F2-Score** (giving more weight to Recall) as primary metrics, with Precision, Recall, and F1-Score as supporting metrics.

## 2. Data Sources
- [Dataset 1](data/raw/data_telco_customer_churn.csv) - Raw Dataset of Telco Customer Churn
- [Dataset 2](data/interim/data_clean.csv) – Clean Dataset of Telco Customer Churn
- [Dataset 3](data/processed/train_data.csv) – Train Dataset of Telco Customer Churn
- [Dataset 4](data/processed/test_data.csv) – Test Dataset of Telco Customer Churn

**Attribute Information**

| Column | Data Type | Description |
|---|---|---|
| **Dependents** | Categorical (Binary) | Does the customer have any liability (Yes/No) |
| **tenure** | Numerical (Integer) | Subscription period in months (0–72) |
| **OnlineSecurity** | Categorical | Online Security service status (Yes/No/No internet service) |
| **OnlineBackup** | Categorical | Online Backup service status (Yes/No/No internet service) |
| **InternetService** | Categorical | Type of internet service (DSL/Fiber optic/No) |
| **DeviceProtection** | Categorical | Device Protection Status (Yes/No/No internet service) |
| **TechSupport** | Categorical | Tech Support service status (Yes/No/No internet service) |
| **Contract** | Categorical (Ordinal) | Contract type (Month-to-month/One year/Two year) |
| **PaperlessBilling** | Categorical (Binary) | Does the customer uses paperless billing (Yes/No) |
| **MonthlyCharges** | Numerical (Float) | Monthly bill in USD (18.8–118.65) |
| **Churn** | Categorical (Target) | Is customer churn (Yes/No) |

## 3. Technologies Used
- Programming Language: Python (e.g., Pandas, NumPy)
- Visualization: Matplotlib, Seaborn, Plotly
- Version Control: Git
- Others: Jupyter Notebook

## 4. Project Structure

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed/cleaned.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.

```

## 5. Summary of Finding
### 5.1 Conclusion
**1. About the Dataset:**
- The telco churn dataset consists of 4,930 customers with 11 features, of which 26.7% of customers churn. Class imbalance (ratio ~2.75:1) needs to be explicitly addressed.
- There are no missing values, but 77 duplicate records have been removed.

**2. Key Factors Causing Churn:**
- **Month-to-Month Contracts** → churn rate of ~43%, the highest compared to other contract types
- **Short Tenure** → New customers (0-12 months) are much more susceptible to churn
- **Fiber Optic Services** → churn rate of ~42%, likely due to higher prices
- **Lack of additional services** (Online Security, Tech Support) → higher churn
- **High Monthly Charges** → Customers with large bills are more likely to churn if the value received is not commensurate

**3. Model Performance:**
- The best model is able to identify churned customers with a high F2-Score
- The combination of SMOTE + Hyperparameter Tuning + Threshold Optimization significantly improved Recall and F2-Score
- Threshold was optimized to maximize the F2-Score (greater weight on Recall), because False Negatives (undetected churned customers) are more detrimental to the company


### 5.2 Business Recommendations
| Priority | Recommendations | Target Segment |
|---|---|---|
| **High** | Special retention program for Month-to-Month customers | Monthly contract customers |
| **High** | Contract upgrade incentive (annual/two-year discount) | Tenure <12 months |
| Medium | Bundled package of additional services (TechSupport + Security) | Users without add-ons |
| Medium | Loyalty & reward program for existing customers | Tenure > 24 months |
| Long Term | Review fiber optic pricing vs. competitors | Fiber optic users |
| Long Term | Early warning system using this ML model | Entire customer base |


### 5.3 Technical Recommendations
1. **Add new features**: Customer satisfaction score, complaint frequency, payment history, and NPS score to improve predictions.
2. **Model monitoring**: Re-evaluate quarterly as customer behavior can change.
3. **A/B testing**: Test the effectiveness of retention interventions on customers predicted to churn.
4. **Business threshold**: Adjust the threshold based on retention cost vs. customer lifetime value.

## 6. Contact
- Name: Muhammad Ilman Mughni
- Email: ilmanmughni29@gmail.com
- Linkedin: https://www.linkedin.com/in/milmanmughni/
