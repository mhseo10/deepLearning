import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
data = np.loadtxt('..\\Data\\data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

# Model
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train
model.fit(x_data, y_data, epochs=1000)
model.summary()
