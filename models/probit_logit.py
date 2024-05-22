# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the dataset
# Assume the file is named 'mushroom.csv' and located in the same directory as this script
data = pd.read_csv('../data/mushroom_cleaned.csv')

# Preprocess the data
# Convert categorical variables to numerical values using one-hot encoding
data = pd.get_dummies(data)

# Separate features and target label
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (Logit Model)
logit_model = LogisticRegression(random_state=42, max_iter=10000)
logit_model.fit(X_train, y_train)
logit_pred = logit_model.predict(X_test)

# Evaluate the Logit model
logit_accuracy = accuracy_score(y_test, logit_pred)
print(f'Logit Model Accuracy: {logit_accuracy:.2f}')

logit_cm = confusion_matrix(y_test, logit_pred)
print('Logit Model Confusion Matrix:')
print(logit_cm)

logit_cr = classification_report(y_test, logit_pred)
print('Logit Model Classification Report:')
print(logit_cr)

# Probit Regression
probit_model = sm.Probit(y_train, sm.add_constant(X_train)).fit()
probit_pred_prob = probit_model.predict(sm.add_constant(X_test))
probit_pred = (probit_pred_prob > 0.5).astype(int)

# Evaluate the Probit model
probit_accuracy = accuracy_score(y_test, probit_pred)
print(f'Probit Model Accuracy: {probit_accuracy:.2f}')

probit_cm = confusion_matrix(y_test, probit_pred)
print('Probit Model Confusion Matrix:')
print(probit_cm)

probit_cr = classification_report(y_test, probit_pred)
print('Probit Model Classification Report:')
print(probit_cr)

# Visualize the coefficients of both models
plt.figure(figsize=(12, 6))

# Logistic Regression Coefficients
logit_coefs = pd.Series(logit_model.coef_[0], index=X.columns)
logit_coefs.sort_values().plot(kind='bar', color='blue', alpha=0.7)
plt.title('Logit Model Coefficients')
plt.savefig('results/logit_model_coefficients.pdf')
plt.show()

# Probit Regression Coefficients
plt.figure(figsize=(12, 6))
probit_coefs = pd.Series(probit_model.params[1:], index=X.columns)
probit_coefs.sort_values().plot(kind='bar', color='green', alpha=0.7)
plt.title('Probit Model Coefficients')
plt.savefig('results/probit_model_coefficients.pdf')
plt.show()
