import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, _), (_, _) = mnist.load_data()

# Training Params
training_epochs = 100
batch_size = 100  # 60000 / batch_size 시 나머지가 없도록 설정 (동일한 input_shape)

# Network Params
dim_image = 784
nHL_G = 256
nHL_D = 256
dim_noise = 100

x_train = x_train.reshape([-1, 784]).astype('float32') / 255.

# batch
train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(60000).batch(batch_size)


# A custom initialization (Xavier Glorot init)
@tf.function
def glorot_init(shape):
    return tf.random.normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


W = {
    'HL_G': tf.Variable(glorot_init([dim_noise, nHL_G])),
    'OL_G': tf.Variable(glorot_init([nHL_G, dim_image])),
    'HL_D': tf.Variable(glorot_init([dim_image, nHL_D])),
    'OL_D': tf.Variable(glorot_init([nHL_D, 1]))
}

b = {
    'HL_G': tf.Variable(tf.zeros([nHL_G])),
    'OL_G': tf.Variable(tf.zeros([dim_image])),
    'HL_D': tf.Variable(tf.zeros([nHL_D])),
    'OL_D': tf.Variable(tf.zeros([1]))
}

train_g = tf.optimizers.Adam(0.0002)
train_d = tf.optimizers.Adam(0.0002)

vars_g = [W['HL_G'], W['OL_G'], b['HL_G'], b['OL_G']]
vars_d = [W['HL_D'], W['OL_D'], b['HL_D'], b['OL_D']]


# Neural Network: Generator
@tf.function
def nn_gen(x):
    hl = tf.nn.relu(tf.add(tf.matmul(x, W['HL_G']), b['HL_G']))
    ol = tf.nn.sigmoid(tf.add(tf.matmul(hl, W['OL_G']), b['OL_G']))  # output format 을 0~1 사이로 만들기 위해 사용
    return ol


# Neural Network: Discriminator
@tf.function
def nn_dis(x):
    hl = tf.nn.relu(tf.add(tf.matmul(x, W['HL_D']), b['HL_D']))
    ol = tf.nn.sigmoid(tf.add(tf.matmul(hl, W['OL_D']), b['OL_D']))
    return ol


def optimization(x_data):
    noise = np.random.uniform(-1., 1., size=[batch_size, dim_noise]).astype(np.float32)

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        sample_g = nn_gen(noise)
        d_fake = nn_dis(sample_g)
        d_real = nn_dis(x_data)

        loss_g = -tf.reduce_mean(tf.math.log(d_fake))
        loss_d = -tf.reduce_mean(tf.math.log(d_real) + tf.math.log(1. - d_fake))

    gradients_g = tape1.gradient(loss_g, vars_g)
    train_g.apply_gradients(zip(gradients_g, vars_g))

    gradients_d = tape2.gradient(loss_d, vars_d)
    train_d.apply_gradients(zip(gradients_d, vars_d))


for epoch in range(training_epochs):
    print(epoch)

    for batch_x in train_data:
        optimization(batch_x)

    if epoch % 10 == 0:
        f, a = plt.subplots(4, 10, figsize=(10, 4))

        for i in range(10):
            z = np.random.uniform(-1., 1., size=[4, dim_noise]).astype(np.float32)
            g = nn_gen(z)
            g = np.reshape(g, newshape=(4, 28, 28, 1))
            g = -1 * (g - 1)

            for j in range(4):
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                a[j][i].imshow(img)

        f.show()
        plt.draw()
