# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model, to_categorical

# Load the dataset
# Assume the file is named 'mushroom.csv' and located in the same directory as this script
data = pd.read_csv('../data/mushroom_cleaned.csv')

# Preprocess the data
# Convert categorical variables to numerical values using one-hot encoding
  = pd.get_dummies(data)

# Separate features and target label
X = data.drop('class', axis=1)
y = data['class']

# One-hot encode the target variable
y = to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
# Define the neural network model
model = Sequential([
    Dense(16, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')  # Output layer with softmax activation for two classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Neural Network Accuracy: {accuracy:.2f}')

# Predict on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob.argmax(axis=1))

# Convert y_test back to single column for evaluation
y_test_single = y_test.argmax(axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_test_single, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
cr = classification_report(y_test_single, y_pred)
print('Classification Report:')
print(cr)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('results/nn_accuracy.pdf')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('results/nn_loss.pdf')
plt.show()

# Save model architecture as a PDF
# plot_model(model, to_file='nn_model_architecture.pdf', show_shapes=True, show_layer_names=True)
