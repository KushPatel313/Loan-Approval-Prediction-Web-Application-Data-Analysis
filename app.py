import streamlit as st
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and scaler
models = {
    'Logistic Regression': pk.load(open('/Users/kushpatel/Downloads/Kush developing/LogisticRegression.pkl', 'rb')),
    'Decision Tree': pk.load(open('/Users/kushpatel/Downloads/Kush developing/DecisionTree.pkl', 'rb')),
    'Random Forest': pk.load(open('/Users/kushpatel/Downloads/Kush developing/RandomForest.pkl', 'rb')),
    'SVM': pk.load(open('/Users/kushpatel/Downloads/Kush developing/SVM.pkl', 'rb'))
}

# Accuracy values for each model (you can update these based on actual model evaluations)
model_accuracies = {
    'Logistic Regression': 0.85,  # Example accuracy values
    'Decision Tree': 0.78,
    'Random Forest': 0.90,
    'SVM': 0.88
}

scaler = pk.load(open('/Users/kushpatel/Downloads/Kush developing/scaler.pkl', 'rb'))
scaler_feature_names = scaler.get_feature_names_out()

st.header('Loan Approval Prediction App')

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

# Session state for cleaned data, predictions, reasons, and suggestions
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = None
if 'reasons' not in st.session_state:
    st.session_state.reasons = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()
    st.write("Uploaded Data Preview:", data.head())

    # Data Cleaning
    if st.button("Run Data Cleaning"):
        data.drop(columns=['loan_id'], errors='ignore', inplace=True)
        data['education'] = data['education'].str.strip().replace(['Graduate', 'Not Graduate'], [1, 0])
        data['self_employed'] = data['self_employed'].str.strip().replace(['No', 'Yes'], [0, 1])
        data['loan_status'] = data['loan_status'].str.strip().replace(['Approved', 'Rejected'], [1, 0])
        data['Assets'] = data[['residential_assets_value', 'commercial_assets_value', 
                               'luxury_assets_value', 'bank_asset_value']].sum(axis=1)
        st.session_state.cleaned_data = data

    if st.session_state.cleaned_data is not None:
        st.write("Cleaned Data Preview:")
        st.dataframe(st.session_state.cleaned_data)

    # User Inputs
    no_of_dep = st.slider('Choose Number of Dependents', 0, 5)
    grad = st.selectbox('Choose Education Level', ['Graduated', 'Not Graduated'])
    self_emp = st.selectbox('Are you Self-Employed?', ['Yes', 'No'])
    annual_income = st.slider('Annual Income', 0, 10000000)
    loan_amount = st.slider('Loan Amount', 0, 10000000)
    loan_dur = st.slider('Loan Duration (in Years)', 0, 20)
    cibil = st.slider('CIBIL Score', 0, 1000)
    bank_asset_value = st.slider('Bank Asset Value', 0, 10000000)
    commercial_assets_value = st.slider('Commercial Asset Value', 0, 10000000)
    luxury_assets_value = st.slider('Luxury Asset Value', 0, 10000000)
    residential_assets_value = st.slider('Residential Asset Value', 0, 10000000)

    grad_s = 0 if grad == 'Graduated' else 1
    emp_s = 0 if self_emp == 'No' else 1

    model_choice = st.selectbox('Select Prediction Model', list(models.keys()))
    selected_model = models[model_choice]

    def analyze_prediction(data):
        reasons = []
        suggestions = []

        if data['cibil_score'] < 700:
            reasons.append("Low CIBIL Score")
            suggestions.append("Consider increasing your CIBIL score to at least 700.")
        if data['income_annum'] < (data['loan_amount'] * 2):
            reasons.append("Insufficient income relative to loan amount.")
            suggestions.append("Increasing your income or applying for a smaller loan amount may improve approval chances.")
        if data['self_employed'] == 1 and data['income_annum'] < 500000:
            reasons.append("Self-employed status with low income.")
            suggestions.append("Consider providing additional income sources or applying with a co-borrower.")
        if data['loan_term'] < 5:
            reasons.append("Short loan term requested.")
            suggestions.append("A longer loan term might improve affordability.")
        total_assets = data[['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']].sum()
        if total_assets < (data['loan_amount'] * 0.5):
            reasons.append("Low total asset value relative to loan amount.")
            suggestions.append("Increasing your assets or applying for a smaller loan amount could help.")
        if not reasons and not suggestions:
            suggestions.append("Congratulations!! You don't need any suggestions, because you have well-maintained details for Loan approval.")
        return reasons, suggestions

    if st.button("Predict Loan Approval"):
        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, annual_income, loan_amount, loan_dur, cibil, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])
        
        if list(pred_data.columns) == list(scaler_feature_names):
            pred_data_scaled = scaler.transform(pred_data)
            prediction = selected_model.predict(pred_data_scaled)

            # Update prediction logic to reject loans with zero or insufficient income
            if annual_income <= 0 or (annual_income < (loan_amount * 2)):
                prediction[0] = 0  # Force a rejection if income is zero or not sufficient

            if prediction[0] == 1:
                st.session_state.prediction_result = "Loan is Approved!"
                st.session_state.reasons = []
                st.session_state.suggestions = analyze_prediction(pred_data.iloc[0])[1]
            else:
                st.session_state.prediction_result = "Loan is Rejected."
                st.session_state.reasons, st.session_state.suggestions = analyze_prediction(pred_data.iloc[0])

    # Display prediction result
    if st.session_state.prediction_result:
        if st.session_state.prediction_result == "Loan is Approved!":
            st.success(st.session_state.prediction_result)
            st.write("### Improvement Suggestions:")
            for suggestion in st.session_state.suggestions or []:
                st.write(f"- {suggestion}")
        else:
            st.error(st.session_state.prediction_result)
            st.write("### Reasons for Rejection:")
            for reason in st.session_state.reasons or []:
                st.write(f"- {reason}")
            st.write("### Improvement Suggestions:")
            for suggestion in st.session_state.suggestions or []:
                st.write(f"- {suggestion}")

    # Step 3: Visualization and Analysis
    if st.button("Generate Visualizations"):
        st.write("## Summary Statistics:")
        summary_stats = st.session_state.cleaned_data[['no_of_dependents', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']].describe()
        st.write(summary_stats)

        st.write("### Distribution Plots:")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        sns.histplot(st.session_state.cleaned_data['income_annum'], bins=30, ax=axes[0, 0], kde=True).set_title('Annual Income Distribution')
        sns.histplot(st.session_state.cleaned_data['loan_amount'], bins=30, ax=axes[0, 1], kde=True).set_title('Loan Amount Distribution')
        sns.histplot(st.session_state.cleaned_data['cibil_score'], bins=30, ax=axes[1, 0], kde=True).set_title('CIBIL Score Distribution')
        sns.histplot(st.session_state.cleaned_data['loan_term'], bins=30, ax=axes[1, 1], kde=True).set_title('Loan Term Distribution')
        sns.histplot(st.session_state.cleaned_data['no_of_dependents'], bins=30, ax=axes[2, 0], kde=True).set_title('Dependents Distribution')
        sns.histplot(st.session_state.cleaned_data['self_employed'], bins=30, ax=axes[2, 1], kde=True).set_title('Self-Employed Status Distribution')
        plt.tight_layout()
        st.pyplot(fig)

        # Model Comparison
        st.write("### Model Accuracy Comparison:")
        model_names = list(model_accuracies.keys())
        accuracies = list(model_accuracies.values())
        
        comparison_fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=model_names, y=accuracies, ax=ax, palette='viridis')
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        
        # Highlight the selected model's accuracy
        selected_accuracy = model_accuracies[model_choice]
        ax.axhline(selected_accuracy, color='r', linestyle='--', label=f'Selected Model: {model_choice} (Accuracy: {selected_accuracy})')
        ax.legend()
        
        st.pyplot(comparison_fig)

        # Display selected model's accuracy
        st.write(f"### Selected Model Accuracy: {model_choice} - {selected_accuracy:.2f}")
