import tensorflow as tf

# tf.constant() 사용
x_data = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
y_data = tf.constant([[10], [20], [30]], dtype=tf.float32)
x_test = tf.constant([[4, 4]], dtype=tf.float32)

# np.array() 사용
# x_data = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
# y_data = np.array([[10], [20], [30]], dtype=np.float32)
# x_test = np.array([[4, 4]], dtype=np.float32)


W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))


@tf.function
def linear_model(x, w):
    return tf.matmul(x, w) + b


@tf.function
def loss_func(pred, y):
    return tf.reduce_mean(tf.square(pred - y))


def optimization():
    with tf.GradientTape() as tape:
        model = linear_model(x_data, W)
        loss = loss_func(model, y_data)

    gradients = tape.gradient(loss, [W, b])
    tf.keras.optimizers.SGD(0.01).apply_gradients(zip(gradients, [W, b]))

    return loss


# Training
for step in range(2001):
    c = optimization()
    if step % 100 == 0:
        print("step:", step)
        print(c.numpy(), "\n")

# Result
print("W: ", W.numpy())
print("b: ", b.numpy(), "\n")

# Testing
model_test = linear_model(x_test, W)
print("model for [4,4]: ", model_test.numpy())
