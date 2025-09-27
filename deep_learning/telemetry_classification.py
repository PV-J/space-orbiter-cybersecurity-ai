import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Example telemetry dataset: 10 features
x_train = np.random.rand(1000, 10)
# Labels: 0=normal, 1=anomaly
y_train = np.random.randint(2, size=1000)

# Simple feedforward neural network
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Predict anomaly probability for a new sample
sample_input = np.random.rand(1, 10)
prediction = model.predict(sample_input)
print(f'Anomaly probability: {prediction[0][0]:.3f}')
