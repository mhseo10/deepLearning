import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Data
x_data = np.array([[1], [2], [3]], dtype=np.float32)
y_data = np.array([[1], [2], [3]], dtype=np.float32)

# Model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='adam')

# Train
model.fit(x_data, y_data, epochs=1000, verbose=0)
model.summary()

print(model.get_weights())
print(model.predict(np.array([4])))

plt.scatter(x_data, y_data)
plt.plot(x_data, y_data)
plt.grid(True)
plt.show()
