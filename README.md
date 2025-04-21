# 🧠 Customer Segmentation using KNN, Random Forest, and XGBoost

This project performs customer segmentation based on **RFM analysis** (Recency, Frequency, Monetary) and applies **machine learning models** to classify customers into segments: **Low**, **Medium**, and **High** value.

---

## 📊 Objective

To compare the performance of the following classification models:
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

...on predicting customer value segments using scaled RFM features.

---

## 📁 Dataset

- Input data must contain **Recency**, **Frequency**, **Monetary** columns and a categorical `Segment` column (`Low Value`, `Medium Value`, `High Value`).

### Example:
```csv
CustomerID,Recency,Frequency,Monetary,Segment
12345,10,5,300,High Value
67890,60,2,150,Medium Value
