'''
Loan Status Prediction
Author: Henry Ha
'''
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
loan_data = pd.read_csv('loan_data.csv')

#TODO EDA

# Display the first few rows
print(loan_data.head())

# Check for missing values
loan_data.isnull().sum()

# Dataset structure
print(loan_data.info())

# Statistical summary
print(loan_data.describe())

# Plot categorical features
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for feature, ax in zip(categorical_features, axes.flatten()):
    sns.countplot(data=loan_data, x=feature, ax=ax)
    ax.set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Plot numerical features
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for feature in numerical_features:
    plt.figure(figsize=(10, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(loan_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(loan_data[feature])
    plt.title(f'Boxplot of {feature}')

    plt.tight_layout()
    plt.show()

# Check unique values in the 'Dependents' column
print(loan_data['Dependents'].unique())

# Convert 'Dependents' to numerical format
loan_data['Dependents'] = loan_data['Dependents'].replace({'3+': 3})
loan_data['Dependents'] = pd.to_numeric(loan_data['Dependents'], errors='coerce')

# Drop non-numerical or irrelevant columns
numerical_data = loan_data.select_dtypes(include=[np.number])

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#TODO Data Preprocessing

# Fill missing categorical values with mode
for column in ['Gender', 'Married', 'Self_Employed']:
    loan_data[column].fillna(loan_data[column].mode()[0], inplace=True)

# Fill missing numerical values with median
loan_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']] = loan_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']].apply(lambda col: col.fillna(col.median() if col.dtype != 'object' else col.mode()[0]))

# Check for missing values after filling
loan_data.isnull().sum()

# Encode categorical features
loan_data = pd.get_dummies(loan_data, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

loan_data.info()

# Feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Include 'Dependents' in scaling
scaled_features = scaler.fit_transform(loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents']])
scaled_df = pd.DataFrame(scaled_features, columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents'])
loan_data.update(scaled_df)

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

X = loan_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = loan_data['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert target to binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TODO Model Training

# Logistic regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Support vector machine (SVM)
from sklearn.svm import SVC

svm_model = SVC(probability=True)  # Enable probability estimates for ROC curve
svm_model.fit(X_train, y_train)

# Evaluate the models
from sklearn.metrics import classification_report

# Logistic Regression Performance
print("Logistic Regression Performance:")
print(classification_report(y_test, log_reg.predict(X_test)))

# SVM Performance
print("SVM Performance:")
print(classification_report(y_test, svm_model.predict(X_test)))

#TODO Model Fine-Tuning

# Use SMOTE

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Logistic Regression on SMOTE data
log_reg_smote = LogisticRegression()
log_reg_smote.fit(X_train_smote, y_train_smote)

# Evaluate the model
print("Logistic Regression with SMOTE Performance:")
print(classification_report(y_test, log_reg_smote.predict(X_test)))

# Train Logistic Regression with class weighting
log_reg_weighted = LogisticRegression(class_weight='balanced')
log_reg_weighted.fit(X_train, y_train)

# Evaluate the model
print("Logistic Regression with Class Weighting Performance:")
print(classification_report(y_test, log_reg_weighted.predict(X_test)))

