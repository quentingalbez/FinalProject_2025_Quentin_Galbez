# 🎯 Recommendation System Project - Final Project 2025

This project implements a recommendation system using multiple approaches, including the **Two Towers model** and the **Alternating Least Squares (ALS)** algorithm. The goal is to predict user-video interactions and provide personalized recommendations based on user and item features.

---

## 🧱 Project Architecture

```
FinalProject_2025_Quentin_Galbez/
├── data/
│   ├── load_data.py           # Script to load and preprocess data
│   └── data_detail.ipynb      # Notebook for data analysis and visualization
├── models/
│   ├── two_towers.ipynb       # Implementation of the Two Towers model
│   └── als.ipynb              # Implementation of the ALS model
├── requirements.txt           # List of required Python libraries
├── .gitignore                 # Files and folders to ignore in version control
└── README.md                  # Project documentation
```

---

## 📁 File Descriptions

### `data/load_data.py`
Handles data loading and preprocessing:
- Downloads and extracts the dataset.
- Loads CSV files into DataFrames.
- Cleans data (e.g., removes duplicates, handles missing values).
- Returns both raw and cleaned data for analysis and modeling.

---

### `data/data_detail.ipynb`
Provides exploratory data analysis (EDA):
- Explores data structure and distributions.
- Visualizes patterns with Matplotlib and Seaborn.
- Analyzes features like user activity levels and video categories.
- Findings guide feature engineering and model design.

---

### `models/two_towers.ipynb`
Implements the **Two Towers model**, a deep learning approach:
- **User Tower**: Processes user features.
- **Item Tower**: Processes item features.
- **Concatenation Layer**: Merges towers.
- **Output Layer**: Predicts interaction scores.

#### Conclusion:
- R² score: ~0.06 (moderate predictive performance).
- Improvements possible via better feature engineering and tuning.

---

### `models/als.ipynb`
Implements the **ALS model** using PySpark:
- Uses matrix factorization to predict user-item scores.
- Applies regularization and cold-start strategies.

#### Conclusion:
- R² score: ~0.24 (better than Two Towers).
- Good baseline with potential for hybrid enhancements.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
pyspark
```

---

## 🔍 Model Insights

### 🔹 Two Towers Model
- ✅ **Strengths**: Captures nonlinear patterns in user/item features.
- ❌ **Weaknesses**: Susceptible to overfitting, needs better features.

### 🔹 ALS Model
- ✅ **Strengths**: Scalable, robust for dense interaction data.
- ❌ **Weaknesses**: Cold-start problem, linear assumptions.

---

## ▶️ How to Run the Project

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Preprocessing**  
   Execute `load_data.py` to prepare the dataset.

3. **Analyze Data**  
   Open and run `data/data_detail.ipynb`.

4. **Train Models**
   - Run `models/two_towers.ipynb` for the deep learning model.
   - Run `models/als.ipynb` for the ALS model.

5. **Evaluate**  
   Compare RMSE, MAE, and R² between models.

---

## ✅ Conclusion

This project explores two core recommendation approaches:
- A **neural network-based Two Towers model**.
- A **collaborative filtering ALS model**.

While ALS currently outperforms the deep model, future work will focus on:
- Improved features,
- Model optimization,
- Hybrid recommendation strategies.