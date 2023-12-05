import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load your protein structure dataset (features and labels)
dataset = pd.read_csv("raw_dataset/2022-12-17-pdb-intersect-pisces_pc30_r2.5")
dataset.head()
# Preprocess the data
# Split the dataset into training and testing sets

# Build a simple RNN model using TensorFlow/Keras
model = models.Sequential([
    layers.Embedding(input_dim=num_amino_acids, output_dim=embedding_dim, input_length=max_sequence_length),
    layers.SimpleRNN(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Save the trained model
model.save('protein_structure_prediction_rnn_model.h5')