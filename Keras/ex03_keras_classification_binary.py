import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
x_data = np.array([[1, 2], [2, 3], [3, 4], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

# Model
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train
model.fit(x_data, y_data, epochs=10000)
model.summary()

# Test
print(model.get_weights())
print(model.predict(x_data))
