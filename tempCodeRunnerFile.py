import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score
)


# 1.Choose a binary classification dataset.
df = pd.read_csv("data.csv")
# Drop empty or irrelevant columns
df.drop(columns=['Unnamed: 32'], inplace=True, errors='ignore')

print("Dataset Shape: ",df.shape)
print(df.head(10))
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 4. Replace dirty missing indicators with np.nan
df.replace(['?', ' ', '--', 'NA', 'N/A'], np.nan, inplace=True)

# 5. Identify column types
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Fill missing values
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.mean()))

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 7. Final NaN Check
if df.isnull().sum().sum() > 0:
    print("\n❌ Still missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    exit()
else:
    print("\n✅ All missing values cleaned successfully!")

y = df['diagnosis']
x = df.drop(columns=['diagnosis'])

# 2.Train/test split and standardize features.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)


if x_train.isnull().sum().sum() > 0 or x_test.isnull().sum().sum() > 0:
    print("❌ There are still NaNs in the data!")
    print(x_train.isnull().sum())
else:
    print("✅ No missing values remain. Proceeding to training...")

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 3.Fit a Logistic Regression model.
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
y_proba = model.predict_proba(x_test_scaled)[:, 1]




# 4.Evaluate with confusion matrix, precision, reca l, ROC-AUC.
print("\nAccuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC Score: ", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()



# 5.Tune threshold and explain sigmoid function.
custom_threshold = 0.4
y_custom_pred = (y_proba > custom_threshold).astype(int)
print(f"\nConfusion Matrix at threshold {custom_threshold}:\n", confusion_matrix(y_test, y_custom_pred))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Generate values for z (input to sigmoid)
z = np.linspace(-10, 10, 1000)
sigmoid_values = sigmoid(z)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid_values, label='Sigmoid(z)', color='blue')
plt.title("Sigmoid Function Curve")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.axvline(x=0, color='k', linestyle='--')  # Vertical line at z = 0
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold = 0.5')
plt.legend()
plt.show()

df.to_csv("cleaned_data.csv", index=False)
print("📁 Cleaned data saved as 'cleaned_data.csv'")
