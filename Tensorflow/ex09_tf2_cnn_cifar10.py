import tensorflow as tf
import time
from keras.datasets import cifar10
from keras.utils import np_utils

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

training_epochs = 15
batch_size = 32

# Shape
print('train_images:', train_images.shape)
print('train_labels:', train_labels.shape)
print('test_images:', test_images.shape)
print('test_labels:', test_labels.shape)

train_images = train_images.reshape([-1, 32, 32, 3]).astype('float32') / 255.
test_images = test_images.reshape([-1, 32, 32, 3]).astype('float32') / 255.

# One-Hot Encoding
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

# batch
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_data = train_data.shuffle(50000).batch(batch_size)


class ConvNet(tf.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layer : shape=[filter height, filter width, channel, output]
        self.W1 = tf.Variable(tf.random.normal(shape=[3, 3, 3, 32], stddev=0.01))  # input = 컬러 채널이므로 3
        self.W2 = tf.Variable(tf.random.normal(shape=[3, 3, 32, 64], stddev=0.01))

        # FC Layer
        self.W3 = tf.Variable(tf.random.normal(shape=[8 * 8 * 64, 10], stddev=0.01))
        self.b3 = tf.Variable(tf.random.normal(shape=[10]))

    def __call__(self, x):
        # Convolution Layer1 32@ 32 * 32
        conv1 = tf.nn.conv2d(x, self.W1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)

        # Pooling Layer1 32@ 16 * 16
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution Layer2 64@ 16 * 16
        conv2 = tf.nn.conv2d(pool1, self.W2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)

        # Pooling Layer2 64@ 8 * 8
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully Connected(FC) Layer
        linear_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        model_linear = tf.matmul(linear_flat, self.W3) + self.b3

        return model_linear


cnn = ConvNet()
optimizer = tf.keras.optimizers.Adam(0.01)


@tf.function
def optimization(model, x, y):
    with tf.GradientTape() as tape:
        pred_linear = model(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_linear, labels=y))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Training
t1 = time.time()

for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = int(train_images.shape[0] / batch_size)

    for batch_x, batch_y in train_data:  # x_train, y_train
        optimization(cnn, batch_x, batch_y)
        current_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=cnn(batch_x), labels=batch_y))
        avg_loss += current_loss / total_batch

    if epoch % 1 == 0:
        print("Step: %d, Loss: %f" % (epoch, avg_loss))

t2 = time.time()

# Testing
pred = tf.nn.softmax(cnn(test_images))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(test_labels, 1)), tf.float32))

print('Training Time (Seconds): ', t2 - t1)
print('Accuracy: %f' % accuracy)
