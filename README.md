# 🎥 YouTube Views Prediction Using Machine Learning

## 📌 Project Overview
This project aims to predict YouTube video views using metadata such as title length, publish time, and number of tags.

## 🎯 Objective
Can we predict video popularity using only basic metadata?

## 🚨 Key Result
After proper time-based validation, the model fails to generalize (R² < 0), showing that metadata alone is insufficient.

## 🧠 Key Insights
- Data is highly skewed → log transformation used
- Weak correlation between features and views
- Random split caused misleading high accuracy (data leakage)
- Time-based split revealed true performance

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn

## ❗ Important Finding

The model initially achieved high accuracy (R² ≈ 0.90) due to improper random splitting.

After fixing the split to respect time order:
→ R² dropped below 0

This highlights the importance of avoiding data leakage in machine learning.
