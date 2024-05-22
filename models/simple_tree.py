# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the dataset
# Assume the file is named 'mushroom.csv' and located in the same directory as this script
data = pd.read_csv("../data/mushroom_cleaned.csv")

# Preprocess the data
# Convert categorical variables to numerical values using one-hot encoding
data = pd.get_dummies(data)

# Separate features and target label
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)  # random state is necessary for the greedy algorithm

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Poisonous', 'Edible'])
plt.savefig('results/decision_tree.pdf')
# plt.show()
