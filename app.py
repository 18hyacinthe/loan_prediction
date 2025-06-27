import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.impute import SimpleImputer

# Configuration de la page
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("Loan Approval Prediction App")
st.markdown("This app allows you to explore loan approval data, visualize model performance, and predict loan approval based on input features.")

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset.csv', encoding='ascii', delimiter=',')
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'no_of_dependents': 'dependents',
        'income_annum': 'income_annual',
        'residential_assets_value': 'residential_value',
        'commercial_assets_value': 'commercial_value',
        'luxury_assets_value': 'luxury_value',
        'bank_asset_value': 'bank_value',
    }, inplace=True)
    return df

df = load_data()

# Sidebar pour la navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page", ["Data Exploration", "Model Performance", "Predict Loan Approval"])

# Page 1: Exploration des données
if page == "Data Exploration":
    st.header("Data Exploration")
    
    # Afficher un aperçu des données
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Dataset Shape: {df.shape}")
    
    # Statistiques descriptives
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Visualisations
    st.subheader("Visualizations")
    
    # Histogramme pour une variable numérique
    num_col = st.selectbox("Select a numerical column for histogram", df.select_dtypes(include=[np.number]).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[num_col], kde=True, bins=30, ax=ax)
    ax.set_title(f'Histogram of {num_col}')
    ax.set_xlabel(num_col)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Countplot pour loan_status
    st.subheader("Loan Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='loan_status', data=df, ax=ax, palette='viridis')
    ax.set_title('Loan Status Distribution')
    ax.set_xlabel('Loan Status')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Boxplot pour une variable numérique par loan_status
    st.subheader("Boxplot by Loan Status")
    num_col_box = st.selectbox("Select a numerical column for boxplot", df.select_dtypes(include=[np.number]).columns, key='boxplot')
    fig, ax = plt.subplots()
    sns.boxplot(x='loan_status', y=num_col_box, data=df, ax=ax, palette='viridis')
    ax.set_title(f'{num_col_box} by Loan Status')
    st.pyplot(fig)

# Page 2: Performance du modèle
elif page == "Model Performance":
    st.header("Model Performance")
    
    # Prétraitement des données
    categorical_cols = ['education', 'self_employed']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded['loan_status_encoded'] = pd.factorize(df_encoded['loan_status'])[0]
    
    X = df_encoded.drop(columns=['loan_id', 'loan_status', 'loan_status_encoded'])
    y = df_encoded['loan_status_encoded']
    
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    st.subheader("Model Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    # Matrice de confusion
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Courbe ROC
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    # Importance des features
    st.subheader("Feature Importance")
    coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots()
    coef.plot(kind='barh', ax=ax, color='teal')
    ax.set_title('Feature Importance (Logistic Regression Coefficients)')
    st.pyplot(fig)

# Page 3: Prédiction
elif page == "Predict Loan Approval":
    st.header("Predict Loan Approval")
    
    # Prétraitement pour la prédiction (MÊME PIPELINE que dans Model Performance)
    categorical_cols = ['education', 'self_employed']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded['loan_status_encoded'] = pd.factorize(df_encoded['loan_status'])[0]
    
    X = df_encoded.drop(columns=['loan_id', 'loan_status', 'loan_status_encoded'])
    y = df_encoded['loan_status_encoded']
    
    # Sauvegarder les noms des colonnes pour la prédiction
    training_columns = X.columns.tolist()
    
    # Imputation et standardisation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Entraînement du modèle
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
      # Formulaire d'entrée
    st.subheader("Enter Loan Details")
    with st.form("prediction_form"):
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=0)
        education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
        income_annual = st.number_input("Annual Income", min_value=0, value=5000000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=10000000)
        loan_term = st.number_input("Loan Term (Years)", min_value=0, value=10)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
        residential_value = st.number_input("Residential Assets Value", min_value=-100000, value=5000000)
        commercial_value = st.number_input("Commercial Assets Value", min_value=0, value=5000000)
        luxury_value = st.number_input("Luxury Assets Value", min_value=0, value=10000000)
        bank_value = st.number_input("Bank Assets Value", min_value=0, value=5000000)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Créer un DataFrame avec les entrées (même structure que les données originales)
            input_df = pd.DataFrame({
                'loan_id': [0],
                'dependents': [dependents],
                'education': [f' {education}'],  # Ajouter l'espace comme dans le CSV original
                'self_employed': [f' {self_employed}'],  # Ajouter l'espace comme dans le CSV original
                'income_annual': [income_annual],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_value': [residential_value],
                'commercial_value': [commercial_value],
                'luxury_value': [luxury_value],
                'bank_value': [bank_value]
            })
            
            # Appliquer le même encodage que pour l'entraînement
            input_encoded = pd.get_dummies(input_df, columns=['education', 'self_employed'], drop_first=True)
            input_encoded = input_encoded.drop(columns=['loan_id'])
            
            # S'assurer que toutes les colonnes d'entraînement sont présentes
            for col in training_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Réordonner les colonnes pour correspondre à l'ordre d'entraînement
            input_encoded = input_encoded[training_columns]
            
            # Imputation et standardisation
            input_imputed = imputer.transform(input_encoded)
            input_scaled = scaler.transform(input_imputed)
            
            # Prédiction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Afficher le résultat
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success(f"Loan Approved! (Probability: {probability:.2%})")
            else:
                st.error(f"Loan Rejected! (Probability of Approval: {probability:.2%})")

if __name__ == "__main__":
    st.write("Built with Streamlit by [Your Name] for a school project.")