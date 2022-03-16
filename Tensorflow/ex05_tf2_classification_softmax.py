import tensorflow as tf
import numpy as np

x_data = np.array(
    [[1, 2, 1, 1], [2, 1, 3, 2],
     [3, 1, 3, 4], [4, 1, 5, 5],
     [1, 7, 5, 5], [1, 2, 5, 6],
     [1, 6, 6, 6], [1, 7, 7, 7]],
    dtype=np.float32)
y_data = np.array(
    [[0, 0, 1], [0, 0, 1],
     [0, 0, 1], [0, 1, 0],
     [0, 1, 0], [0, 1, 0],
     [1, 0, 0], [1, 0, 0]],
    dtype=np.float32)
W = tf.Variable(tf.random.normal([4, 3]))
b = tf.Variable(tf.random.normal([3]))


def optimization():
    with tf.GradientTape() as tape:
        linear_model = tf.matmul(x_data, W) + b
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_model, labels=y_data))

    gradients = tape.gradient(cost, [W, b])
    tf.keras.optimizers.SGD(0.1).apply_gradients(zip(gradients, [W, b]))


# Training
for step in range(2001):
    optimization()

# Result
model = tf.nn.softmax(tf.matmul(x_data, W) + b)
model_index = tf.argmax(model, 1)
print("Model:", model)
print("Index:", model_index)

accuracy = tf.reduce_mean(tf.cast(tf.equal(model_index, tf.argmax(y_data, 1)), tf.float32))
print('Accuracy: {:.3f} '.format(accuracy))

# Testing
x_test = np.array([[1, 8, 8, 8]], dtype=np.float32)
model_test = tf.argmax(tf.nn.softmax(tf.matmul(x_test, W) + b), 1)

if model_test.numpy() == 0:
    print("[1,8,8,8] is A.")
elif model_test.numpy() == 1:
    print("[1,8,8,8] is B.")
elif model_test.numpy() == 2:
    print("[1,8,8,8] is C.")
