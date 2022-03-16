from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, pooling, Flatten, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Shape
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)

x_train = x_train. astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='SAME', strides=1, activation='relu'))
model.add(pooling.MaxPooling2D(2, strides=2))
model.add(Conv2D(64, kernel_size=3, padding='SAME', strides=1, activation='relu'))
model.add(pooling.MaxPooling2D(2, strides=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=15, batch_size=32)
_, accuracy = model.evaluate(x_test, y_test)

print("Accuracy:", accuracy)
model.summary()
