# 📊 **LOAN APPROVAL PREDICTION PROJECT**

A complete machine learning project to predict bank loan approval using Python, Streamlit, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)

## 🎯 **PROJECT OBJECTIVE**

This project aims to develop a machine learning model capable of automatically predicting whether a bank loan will be approved or rejected, based on the applicant's financial and personal characteristics.

## 📋 **PROJECT STRUCTURE**

```
machine-learning/
├── app.py                          # Streamlit web application
├── ml_loan.ipynb                   # Jupyter notebook for analysis
├── loan_approval_dataset.csv       # Dataset (4271 loans)
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── .gitignore                      # Files to ignore
└── machine-learning.code-workspace # VS Code workspace
```

## 🗃️ **DATASET ANALYSIS**

### **Input Variables (Features):**
- **`loan_id`**: Unique loan identifier
- **`no_of_dependents`**: Number of dependents (0-5)
- **`education`**: Education level (Graduate/Not Graduate)
- **`self_employed`**: Employment status (Yes/No)
- **`income_annum`**: Annual income
- **`loan_amount`**: Requested loan amount
- **`loan_term`**: Loan duration in years
- **`cibil_score`**: Credit score (300-900)
- **`residential_assets_value`**: Residential real estate value
- **`commercial_assets_value`**: Commercial assets value
- **`luxury_assets_value`**: Luxury assets value
- **`bank_asset_value`**: Bank assets value

### **Target Variable:**
- **`loan_status`**: Approval status (Approved/Rejected)

**📊 Dataset dimensions:** 4271 rows × 13 columns

## 🔬 **ANALYSIS AND MODELING**

### **1. Exploratory Data Analysis (EDA)**
- **Variable distribution**: Histograms and density plots
- **Correlations**: Correlation matrix with heatmap
- **Relationships**: Pairplots and boxplots by loan status
- **Categorical variables**: Frequency analysis

### **2. Data Preparation**
- **Cleaning**: Removing spaces from column names
- **Missing values handling**: Median imputation
- **Encoding**: One-hot encoding for categorical variables
- **Standardization**: StandardScaler to normalize scales

### **3. Modeling**
- **Algorithm chosen**: Logistic Regression
- **Justification**:
  - Suitable for binary classification
  - Interpretable (explanatory coefficients)
  - Fast and efficient
  - Satisfactory performance

### **4. Model Evaluation**
- **Metrics used**:
  - Accuracy (Overall precision)
  - Confusion matrix
  - ROC curve and AUC
  - Classification report
- **Validation**: 80/20 split (training/test)

## 🚀 **STREAMLIT APPLICATION**

The interactive web application includes **3 main pages**:

### **📊 Page 1: Data Exploration**
- Dataset overview with descriptive statistics
- Interactive histograms for each variable
- Loan status distribution
- Comparative boxplots by approval status

### **📈 Page 2: Model Performance**
- Complete machine learning pipeline
- Detailed performance metrics
- Visualizations: confusion matrix, ROC curve
- Feature importance (regression coefficients)

### **🎯 Page 3: Interactive Prediction**
- Input form for new loans
- Real-time prediction
- Result display with probability

## 🛠️ **INSTALLATION AND SETUP**

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **1. Clone the project**
```bash
git clone https://github.com/your-username/machine-learning.git
cd machine-learning
```

### **2. Create virtual environment**

#### **On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate
```

#### **On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the application**
```bash
streamlit run app.py
```

### **5. Use Jupyter Notebook (optional)**
```bash
# Install Jupyter if not already done
pip install jupyter

# Launch Jupyter
jupyter notebook ml_loan.ipynb
```

The application will automatically open in your browser at: `http://localhost:8501`

## 📱 **USAGE**

1. **Exploration**: Navigate to "Data Exploration" to analyze the dataset
2. **Performance**: Check "Model Performance" to see model metrics
3. **Prediction**: Use "Predict Loan Approval" to test new cases

### **Prediction example:**
```
Number of dependents: 2
Education: Graduate
Self-employed: No
Annual income: 5000000
Loan amount: 10000000
Loan term: 10 years
CIBIL score: 700
...
```

## 🔧 **TECHNOLOGIES USED**

- **Python 3.8+**: Main language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Visualizations
- **Jupyter Notebook**: Exploratory analysis

## 📊 **RESULTS**

- **Model**: Logistic Regression
- **Performance**: [To be completed with your results]
- **Accuracy**: [To be completed]
- **AUC**: [To be completed]

## 🚀 **FUTURE IMPROVEMENTS**

1. **Cross-validation** for more robust evaluation
2. **Feature engineering** (ratios creation, transformations)
3. **Other algorithms** (Random Forest, XGBoost, SVM)
4. **Hyperparameter tuning** to optimize performance
5. **Cloud deployment** (Heroku, AWS, Azure)
6. **REST API** for integration with other systems

## 📞 **CONTACT**

- **Author**: [Your Name]
- **Email**: [kagbedjinou@yahoo.com]
- **GitHub**: [https://github.com/18hyacinthe]
- **Project**: Made as part of a school project


---

⭐ **Don't hesitate to star this project if it was useful to you!**
