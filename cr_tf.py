import numpy as np
import tensorflow as tf
from CSVParse import CSVParse

csv = CSVParse()
csv.read('currency.csv')

data = np.array(csv.data[1:])

diff = []
for i in xrange(1,len(data)):
    diff.append(data[i] - data[i-1])

#target currency: USD, JPY, CNY, EUR, GBP, CHF, CAD, AUD
target = 'EUR'
score = []
for i in xrange(len(diff)-1):
    x = [1,0] if (diff[i+1][csv.data[0].index(target)]>0) else [0,1]
    score.append(x)

### generate data sets
DATA_LEN = len(score)
DATA_RATIO = 0.8
split = int(DATA_LEN * DATA_RATIO)
train_x = diff[:split]
train_y = score[:split]
test_x = diff[split:DATA_LEN]
test_y = score[split:DATA_LEN]

MAX_STEPS = 5000
LEARNING_RATE = 0.005

### select running device cpu:0 or gpu:0
with tf.device('/gpu:0'):

    ### define placeholders for data set
    x = tf.placeholder(tf.float32, [None, 8])
    y = tf.placeholder(tf.float32, [None, 2])

    ### define neural network form
    w_1 = tf.Variable(tf.random_normal([8, 50]))
    b_1 = tf.Variable(tf.zeros([50]))

    w_2 = tf.Variable(tf.random_normal([50,50]))
    b_2 = tf.Variable(tf.zeros([50]))

    w_3 = tf.Variable(tf.random_normal([50,50]))
    b_3 = tf.Variable(tf.zeros([50]))

    w_out = tf.Variable(tf.random_normal([50, 2]))
    b_out = tf.Variable(tf.zeros([2]))

    h_1 = tf.add(tf.matmul(x, w_1), b_1)
    h_1 = tf.nn.relu(h_1)
    h_1 = tf.nn.dropout(h_1, 0.99)

    h_2 = tf.add(tf.matmul(h_1, w_2), b_2)
    h_2 = tf.nn.relu(h_2)
    h_2 = tf.nn.dropout(h_2, 0.99)

    h_3 = tf.add(tf.matmul(h_2, w_3), b_3)
    h_3 = tf.nn.relu(h_3)
    h_2 = tf.nn.dropout(h_3, 0.99)

    out = tf.add(tf.matmul(h_3, w_out), b_out)

    ### define cost function and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

    ### define test values
    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### create session and initialize variables
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

### train model
for step in range(MAX_STEPS):
    _, loss = sess.run([opt, loss_op], feed_dict={x:train_x, y:train_y})
    if (step+1) % 100 == 0:
        print("[{}/{}] loss:{:.3f}".format(step+1, MAX_STEPS, loss))
print("--- Training Finished ---")

### test model
print("Train Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:train_x, y:train_y})))
print("Test  Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:test_x, y:test_y})))

