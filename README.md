# TASK4-Classification_with_Logistic_Regression


This project demonstrates binary classification using **Logistic Regression** on the Breast Cancer Wisconsin dataset.

## 📌 Objective

- To build a binary classifier using logistic regression.
- Evaluate it using standard classification metrics.
- Visualize model performance and the sigmoid curve.
- Handle missing values properly before model training.

---

## 🔧 Tools & Libraries Used

- pandas, numpy
- seaborn, matplotlib
- scikit-learn

---

## 📂 Dataset

Used Breast Cancer Wisconsin dataset from `data.csv`. The target variable is `diagnosis`:
- `M` (Malignant) → `1`
- `B` (Benign) → `0`

---

## ✅ Data Preprocessing

- Mapped target values to binary.
- Replaced dirty missing indicators (e.g., `"?"`, `" "`, `"NA"`) with NaN.
- Handled missing values:
  - **Numerical** → filled with **mean**.
  - **Categorical** → filled with **mode**.
- Dropped irrelevant column: `Unnamed: 32`.

---

## 🧠 Model Training

- Standardized features using `StandardScaler`.
- Trained Logistic Regression model with `max_iter=1000`.
- Evaluated model on 20% test data.

---

## 📊 Results

### Default Threshold (0.5)

**Confusion Matrix**:
```
[[70  1]
 [ 2 41]]
```

**ROC-AUC Score**:  
```
0.99737962659679
```

---

### Custom Threshold (0.4)

**Confusion Matrix at threshold 0.4**:
```
[[70  1]
 [ 1 42]]
```

---

## 🧮 Sigmoid Curve

The sigmoid function was plotted to visualize how logistic regression maps input values (`z`) to probabilities between 0 and 1.
## sigmoid.png ##
---

## 💾 Output

📁 **Cleaned data saved as** `cleaned_data.csv`

---

## 📈 ROC Curve

ROC Curve was plotted using `fpr`, `tpr` from `roc_curve()`.
## ROC.png ##