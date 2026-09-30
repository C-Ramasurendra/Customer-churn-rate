import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"   
os.makedirs("charts", exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
print("Rows, columns:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# STEP 2: Clean the data
# ---------------------------------------------------------------
df = df.drop(columns=["customerID"])                       # ID is not useful for prediction
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks become NaN
print("Missing values before cleaning:", df["TotalCharges"].isna().sum())
df = df.dropna()                                           # remove the few bad rows
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})         # convert Yes/No to 1/0

# ---------------------------------------------------------------
# STEP 3: EDA (exploratory data analysis) - 3 simple charts
# ---------------------------------------------------------------
print("\nChurn rate (%):")
print((df["Churn"].value_counts(normalize=True) * 100).round(1))

# Chart 1: how many customers left vs stayed
df["Churn"].value_counts().rename({0: "Stayed", 1: "Left"}).plot(kind="bar")
plt.title("Customers Who Stayed vs Left")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("charts/1_churn_count.png")
plt.close()

# Chart 2: churn rate by contract type
(df.groupby("Contract")["Churn"].mean() * 100).plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate (%)")
plt.tight_layout()
plt.savefig("charts/2_churn_by_contract.png")
plt.close()

# Chart 3: tenure (months with company) for stayed vs left
df[df["Churn"] == 0]["tenure"].plot(kind="hist", bins=20, alpha=0.6, label="Stayed")
df[df["Churn"] == 1]["tenure"].plot(kind="hist", bins=20, alpha=0.6, label="Left")
plt.title("Tenure of Customers")
plt.xlabel("Months with company")
plt.legend()
plt.tight_layout()
plt.savefig("charts/3_tenure.png")
plt.close()

# ---------------------------------------------------------------
# STEP 4: Prepare data for the models
# ---------------------------------------------------------------
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)  # text columns -> numbers
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------
# STEP 5: Train and compare two models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
    })
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"charts/confusion_{name.replace(' ', '_')}.png")
    plt.close()

results_df = pd.DataFrame(results).round(3)
print("\nModel comparison:")
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------------
# STEP 6: Which factors matter most? (Random Forest)
# ---------------------------------------------------------------
importance = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=False).head(10)
print("\nTop 10 important features:")
print(top10.round(3))

top10.sort_values().plot(kind="barh")
plt.title("Top 10 Factors Affecting Churn")
plt.tight_layout()
plt.savefig("charts/4_feature_importance.png")
plt.close()

print("\nDone! Check the 'charts' folder and model_results.csv")
