import tensorflow as tf


class ANN(object):
    def __init__(self):
        self.W_1 = tf.Variable(tf.random.normal(shape=[784, nH1]))  # H1
        self.W_2 = tf.Variable(tf.random.normal(shape=[nH1, nH2]))  # H2
        self.W_3 = tf.Variable(tf.random.normal(shape=[nH2, nH3]))  # H3
        self.W_Out = tf.Variable(tf.random.normal(shape=[nH3, 10]))  # Output

        self.b_1 = tf.Variable(tf.random.normal(shape=[nH1]))  # H1 bias
        self.b_2 = tf.Variable(tf.random.normal(shape=[nH2]))  # H2 bias
        self.b_3 = tf.Variable(tf.random.normal(shape=[nH3]))  # H3 bias
        self.b_Out = tf.Variable(tf.random.normal(shape=[10]))  # Output bias

    def __call__(self, x):
        h1_out = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
        h2_out = tf.nn.relu(tf.matmul(h1_out, self.W_2) + self.b_2)
        h3_out = tf.nn.relu(tf.matmul(h2_out, self.W_3) + self.b_3)
        out = tf.matmul(h3_out, self.W_Out) + self.b_Out
        return out


batch_size = 128
nH1 = 256
nH2 = 256
nH3 = 256

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# shape 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, sep="\n")

# 28 * 28 2차원 이미지를 1차원으로 조정 후, 데이터 값을 0 ~ 1 사이의 실수로 변환
x_train = x_train.reshape([-1, 784]).astype('float32') / 255.
x_test = x_test.reshape([-1, 784]).astype('float32') / 255.

# one hot encoding
y_train = tf.one_hot(y_train, depth=10)  # 0: [1, 0, ... , 0, 0], 1: [0, 1, ... , 0, 0], ...
y_test = tf.one_hot(y_test, depth=10)

# batch
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # (60000, 784)를 60000개의 slices 로 분리
train_data = train_data.shuffle(60000).batch(batch_size)  # 무작위로 섞은 다음 batch_size 만큼 추출

ANN_model = ANN()
optimizer = tf.optimizers.Adam(0.01)


@tf.function
def optimization(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))


for epoch in range(20):
    avg_loss = 0
    tot_batch = int(x_train.shape[0] / batch_size)

    for batch_x, batch_y in train_data:  # x_train, y_train
        optimization(ANN_model, batch_x, batch_y)
        current_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=ANN_model(batch_x), labels=batch_y))
        avg_loss += current_loss / tot_batch

    if epoch % 1 == 0:
        print("Step: %d, Loss: %f" % (epoch, avg_loss))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ANN_model(x_test), 1), tf.argmax(y_test, 1)), tf.float32))
print("Accuracy: %f" % accuracy)
