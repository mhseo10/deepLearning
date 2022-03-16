from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, sep="\n")

x_train = x_train.reshape([-1, 784]).astype('float32') / 255.
x_test = x_test.reshape([-1, 784]).astype('float32') / 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
_, accuracy = model.evaluate(x_test, y_test)  # return cost, accuracy
print('Accuracy:', accuracy)
model.summary()
