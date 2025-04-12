# 1. Import Libraries
import numpy as np
import pandas as pd

df=pd.read_csv("/content/Online Retail.csv")
df
# Basic exploration
shape = df.shape
columns = df.columns.tolist()
head = df.head()
print(df.info())
print(df.describe(include='all'))
missing_values = df.isnull().sum()

shape, columns, head, missing_values

# data cleaning
# Step 2: Data Cleaning

# Drop rows with missing CustomerID or Description (needed for supervised learning later)
df_clean = df.dropna(subset=['CustomerID', 'Description'])

# Remove rows with Quantity <= 0 or UnitPrice <= 0 (invalid transactions)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Convert CustomerID to string (treat as categorical)
df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)

# Preview the cleaned dataset
clean_shape = df_clean.shape
clean_head = df_clean.head()
clean_missing = df_clean.isnull().sum()

clean_shape, clean_head, clean_missing

# === 1. Import libraries ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 4. Data Cleaning ===
df = df.dropna(subset=['CustomerID', 'Description'])                         # Drop missing IDs/descriptions
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]                        # Keep only positive Quantity/Price
df['CustomerID'] = df['CustomerID'].astype(str)                              # CustomerID as string
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']                          # Add TotalPrice

# === 5. Quantity Distribution ===
sns.histplot(df['Quantity'], bins=50, kde=True)
plt.title('Quantity Distribution')
plt.xlim(0, 100)
plt.show()

# === 6. Unit Price Distribution (Clipped at 100) ===
sns.histplot(df['UnitPrice'].clip(upper=100), bins=50, kde=True)
plt.title('Unit Price Distribution (Clipped at 100)')
plt.show()

# === 7. Top 10 Countries by Transactions ===
top_countries = df['Country'].value_counts().head(10).reset_index()
top_countries.columns = ['Country', 'Transactions']
sns.barplot(data=top_countries, x='Transactions', y='Country', palette='viridis', legend=False)
plt.title('Top 10 Countries by Transactions')
plt.xlabel('Transactions')
plt.ylabel('Country')
plt.show()

# === 8. Top 10 Selling Products ===
top_products = df['Description'].value_counts().head(10).reset_index()
top_products.columns = ['Product', 'Quantity']
sns.barplot(data=top_products, x='Quantity', y='Product', palette='magma', legend=False)
plt.title('Top 10 Selling Products')
plt.xlabel('Quantity')
plt.ylabel('Product')
plt.show()

# === 9. Top 10 Customers by Revenue ===
customer_spending = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
customer_spending.columns = ['CustomerID', 'Revenue']
sns.barplot(data=customer_spending, x='Revenue', y='CustomerID', palette='coolwarm', legend=False)
plt.title('Top 10 Customers by Revenue')
plt.xlabel('Revenue')
plt.ylabel('Customer ID')
plt.show()

# === 10. Correlation Matrix ===
corr = df[['Quantity', 'UnitPrice', 'TotalPrice']].corr()
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()



import datetime as dt

# Reload InvoiceDate index if necessary
df.reset_index(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Set snapshot date (next day after last invoice)
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by CustomerID and compute RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalPrice': 'sum'                                       # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Show top rows
print(rfm.head())

# Use pd.qcut for Recency (lower is better)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=3, labels=[3, 2, 1])

# Use pd.cut for Frequency & Monetary (higher is better)
rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=3, labels=[1, 2, 3])
rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=3, labels=[1, 2, 3])

# Convert scores to integer
rfm[['R_Score', 'F_Score', 'M_Score']] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int)

# Combine into RFM Score
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

# Segment customers based on RFM_Score
rfm['Segment'] = pd.qcut(rfm['RFM_Score'], q=3, labels=['Low Value', 'Medium Value', 'High Value'])

# Final table
rfm_final = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment']]

# Preview
print(rfm_final.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme
sns.set(style="whitegrid")

# 1. 📈 Distribution of RFM Features
plt.figure(figsize=(18, 5))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    plt.subplot(1, 3, i+1)
    sns.histplot(rfm[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 2. 📊 Boxplots to Detect Outliers
plt.figure(figsize=(18, 5))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=rfm[col], color='lightcoral')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 3. 📉 Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of RFM Features')
plt.tight_layout()
plt.show()

# 4. 🧩 RFM Segments Count
plt.figure(figsize=(6, 4))
sns.countplot(data=rfm_final, x='Segment', hue='Segment', order=['High Value', 'Medium Value', 'Low Value'], palette='Set2', legend=False)
plt.title('Customer Segment Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# 5. 🎯 RFM Distribution by Segment (Boxplot with fixed palette warning)
plt.figure(figsize=(18, 5))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=rfm_final, x='Segment', y=col, hue='Segment', order=['High Value', 'Medium Value', 'Low Value'], palette='Set3', legend=False)
    plt.title(f'{col} by Segment')
plt.tight_layout()
plt.show()

# 6. 🧪 Pairplot for RFM Clusters
sns.pairplot(rfm_final, vars=['Recency', 'Frequency', 'Monetary'], hue='Segment', palette='Set1')
plt.suptitle("Pairwise RFM Relationships by Segment", y=1.02)
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Encode target variable
rfm_final['Segment_Label'] = rfm_final['Segment'].map({
    'Low Value': 0,
    'Medium Value': 1,
    'High Value': 2
})

# Define features and target
X = rfm_final[['Recency', 'Frequency', 'Monetary']]
y = rfm_final['Segment_Label']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# ====================== KNN ======================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("KNN Accuracy")
print("Train:", accuracy_score(y_train, knn.predict(X_train)))
print("Test :", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=['Low', 'Medium', 'High']).plot(cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.show()

cv_knn = cross_val_score(knn, X_scaled, y, cv=5)
print("KNN Cross-Validation Scores:", cv_knn)
print("KNN CV Mean Accuracy:", cv_knn.mean())

# ====================== Random Forest ======================
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy")
print("Train:", accuracy_score(y_train, rf.predict(X_train)))
print("Test :", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf, display_labels=['Low', 'Medium', 'High']).plot(cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()

cv_rf = cross_val_score(rf, X_scaled, y, cv=5)
print("Random Forest CV Scores:", cv_rf)
print("RF CV Mean Accuracy:", cv_rf.mean())

# ====================== XGBoost ======================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric='mlogloss',  # Still needed to avoid eval metric warning
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost Accuracy")
print("Train:", accuracy_score(y_train, xgb.predict(X_train)))
print("Test :", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(cm_xgb, display_labels=['Low', 'Medium', 'High']).plot(cmap='Oranges')
plt.title("XGBoost Confusion Matrix")
plt.show()

cv_xgb = cross_val_score(xgb, X_scaled, y, cv=5)
print("XGBoost CV Scores:", cv_xgb)
print("XGB CV Mean Accuracy:", cv_xgb.mean())

import numpy as np

# Accuracy scores
train_accuracies = [
    accuracy_score(y_train, knn.predict(X_train)),
    accuracy_score(y_train, rf.predict(X_train)),
    accuracy_score(y_train, xgb.predict(X_train))
]
test_accuracies = [
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_xgb)
]
models = ['KNN', 'Random Forest', 'XGBoost']

# Plotting with better aesthetics
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='salmon', edgecolor='black')

# Add text labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# Labels and title
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train vs Test Accuracy Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()