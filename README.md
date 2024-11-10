# Loan Approval Prediction Web Application

## Overview
This project is an end-to-end **Loan Approval Prediction Web Application** designed to provide real-time predictions of loan approval status based on applicant data. Built with **Streamlit** for the web interface, this application allows users to upload datasets, perform data cleaning, select machine learning models for prediction, and visualize model performance. The project was designed to showcase predictive modeling, interactive user interfaces, and practical insights into financial risk analysis.

### Key Features
- **Data Upload & Cleaning**: Users can upload custom datasets, which are automatically cleaned and preprocessed.
- **Machine Learning Models**: Four models are integrated—**Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**—enabling users to choose and compare different models.
- **Real-Time Predictions**: Provides predictions for loan approval or rejection with detailed explanations and actionable suggestions.
- **Data Visualization**: Offers summary statistics and visualization of key financial features for insightful analysis.
- **Model Comparison**: Visualizes model accuracy for better understanding of model performance.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Workflow](#detailed-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Feature Analysis](#feature-analysis)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```plaintext
├── Loan_Prediction_App/
│   ├── app.py                       # Streamlit application for loan prediction
│   ├── loan_approval_dataset.csv     # Sample dataset (replace with your dataset)
│   ├── scaler.pkl                    # Scaler object for feature scaling
│   ├── LogisticRegression.pkl        # Trained Logistic Regression model
│   ├── DecisionTree.pkl              # Trained Decision Tree model
│   ├── RandomForest.pkl              # Trained Random Forest model
│   ├── SVM.pkl                       # Trained SVM model
├── Loan_Prediction_Notebook.ipynb    # Jupyter notebook for data preprocessing & training
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

## Dataset
The `loan_approval_dataset.csv` contains data on loan applications, including financial and demographic features, and their approval statuses. Key fields include:
- **Financial Assets**: Bank, residential, commercial, and luxury assets.
- **Applicant Demographics**: Number of dependents, self-employed status, and education level.
- **Loan Features**: Requested loan amount, duration, and applicant's CIBIL score.

**Note**: Remove any identifying information when working with sensitive data.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Loan_Approval_Prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Loan_Approval_Prediction
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the application**:
   ```bash
   streamlit run app.py
   ```
2. **Upload a Dataset**: Click on "Upload your dataset" to import a CSV file.
3. **Data Cleaning**: Click on "Run Data Cleaning" to preprocess the dataset.
4. **Select Prediction Model**: Choose one of the four machine learning models.
5. **Predict Loan Approval**: Adjust parameters or upload data and press "Predict Loan Approval."
6. **Visualization**: Click "Generate Visualizations" to explore dataset distributions and model accuracy.

## Detailed Workflow
1. **Data Cleaning**:
   - **Preprocessing**: Strips whitespace and encodes categorical features like education and employment.
   - **Asset Calculation**: Creates a new feature aggregating different types of assets for a comprehensive view of financial strength.

2. **Feature Engineering**:
   - Converts categorical values (e.g., Education, Self-Employment) to numerical representations.
   - Constructs a total asset value by summing residential, commercial, luxury, and bank assets.

3. **Model Selection and Training**:
   - **Logistic Regression**: Used for binary classification of loan approval.
   - **Decision Tree**: Captures non-linear relationships and provides explainability.
   - **Random Forest**: Reduces overfitting and improves accuracy.
   - **Support Vector Machine (SVM)**: Useful for high-dimensional spaces.

4. **Model Evaluation**:
   - Calculates accuracy scores and compares them across models in the web app.
   - Provides real-time model selection and comparison within the application.

5. **Predictions and Explanations**:
   - Delivers a prediction (approved or rejected) based on user-provided input.
   - Offers suggestions to improve loan approval chances based on input values.

## Machine Learning Models
This project utilizes four machine learning models, each with distinct properties:
- **Logistic Regression**: Ideal for binary classification, providing robust and interpretable results.
- **Decision Tree Classifier**: A non-parametric model that works well for capturing interactions between variables.
- **Random Forest Classifier**: Combines multiple decision trees to improve accuracy and reduce overfitting.
- **Support Vector Machine (SVM)**: Known for handling high-dimensional data effectively and creating robust decision boundaries.

### Model Accuracy Comparison
In the app, accuracy metrics are displayed for each model to help users identify the most reliable prediction model for their needs.

## Feature Analysis
### Key Features for Prediction
- **CIBIL Score**: A critical factor influencing loan approval, typically requiring a score above 700.
- **Income and Loan Amount**: Significant in assessing the applicant’s repayment ability.
- **Total Assets**: High asset values often correlate with higher loan approval rates.

### Suggestions for Improvement
The app provides actionable insights based on prediction outcomes, e.g., improving CIBIL score, increasing income, or applying for a longer loan term.

## Future Improvements
Potential enhancements for this project include:
- **Automated Hyperparameter Tuning**: To improve model accuracy and performance.
- **Advanced Explainability**: Using SHAP or LIME for more granular model interpretability.
- **Real-time Data Integration**: Adding APIs to pull real-time financial or economic data for enhanced predictive power.
- **Expanded Model Options**: Incorporating additional models like Gradient Boosting or XGBoost for comparison.

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request for improvements.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed under the MIT License.
