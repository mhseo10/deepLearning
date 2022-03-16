import tensorflow as tf
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

training_epochs = 15
batch_size = 100

# shape 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, sep="\n")

# 현재 shape이 (60000, 28, 28)로 컬러 채널이 생략되어 있기 때문에 4차원 tensor shape인 (60000, 28, 28, 1)로 재배열
x_train_img = x_train.reshape([-1, 28, 28, 1]).astype('float32') / 255.
x_test_img = x_test.reshape([-1, 28, 28, 1]).astype('float32') / 255.

# one hot encoding
y_train = tf.one_hot(y_train, depth=10)  # 0: [1, 0, ... , 0, 0], 1: [0, 1, ... , 0, 0], ...
y_test = tf.one_hot(y_test, depth=10)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, sep="\n")

# batch
train_data = tf.data.Dataset.from_tensor_slices((x_train_img, y_train))  # (60000, 28, 28, 1)를 60000개의 slices 로 분리
train_data = train_data.shuffle(60000).batch(batch_size)  # 무작위로 섞은 다음 batch_size 만큼 추출


class CNN(tf.Module):
    def __init__(self):
        super().__init__()
        # Convolution Layer : shape=[filter height, filter width, channel, output]
        self.W1 = tf.Variable(tf.random.normal(shape=[3, 3, 1, 32], stddev=0.01))  # input = 흑백 채널이므로 1
        self.W2 = tf.Variable(tf.random.normal(shape=[3, 3, 32, 64], stddev=0.01))  # 이전 레이어의 출력 수 = 채널

        # FC Layer
        self.W3 = tf.Variable(tf.random.normal(shape=[7 * 7 * 64, 10], stddev=0.01))  # 64@ 7 * 7 을 1차원으로 변환
        self.b3 = tf.Variable(tf.random.normal(shape=[10]))

    def __call__(self, x):
        # Convolution Layer1 32@ 28 * 28
        conv1 = tf.nn.conv2d(x, self.W1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)

        # Pooling Layer1 32@ 14 * 14
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution Layer2 64@ 14 * 14
        conv2 = tf.nn.conv2d(pool1, self.W2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)

        # Pooling Layer2 64@ 7 * 7
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully Connected(FC) Layer
        linear_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        model_linear = tf.matmul(linear_flat, self.W3) + self.b3

        return model_linear

    """
        tf.layers 사용: W1, W2는 CL1, CL2로 대체
        layers 사용 시 trainable_variables 로 관리 가능

        # Convolution Layer1
        CL1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', 
                                strides=1, activation=tf.nn.relu)

        # Pooling Layer1
        PL1 = tf.layers.max_pooling2d(inputs=CL1, pool_size=[2, 2], padding='SAME', strides=2)

        # Convolution Layer2
        CL2 = tf.layers.conv2d(inputs=PL1, filters=64, kernel_size=[3, 3], padding='SAME', strides=1,
                                       activation=tf.nn.relu)
        # Pooling Layer2
        PL2 = tf.layers.max_pooling2d(inputs=CL2, pool_size=[2, 2], padding='SAME', strides=2)

        # Fully Connected (FC) Layer
        L_flat = tf.reshape(PL2, [-1, 7 * 7 * 64])
        model_LC = tf.matmul(L_flat, W3) + b3 

    """


CNN_model = CNN()
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
    total_batch = int(x_train.shape[0] / batch_size)

    for batch_x, batch_y in train_data:  # x_train, y_train
        optimization(CNN_model, batch_x, batch_y)
        current_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=CNN_model(batch_x), labels=batch_y))
        avg_loss += current_loss / total_batch

    if epoch % 1 == 0:
        print("Step: %d, Loss: %f" % (epoch, avg_loss))

t2 = time.time()

# Testing
pred = tf.nn.softmax(CNN_model(x_test_img))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1)), tf.float32))

print('Training Time (Seconds): ', t2 - t1)
print('Accuracy: %f' % accuracy)
