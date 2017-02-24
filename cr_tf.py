from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reader import Reader

MAX_STEPS = 10000
BATCH_SIZE = 30
LEARNING_RATE = 0.001 # learning rate for optimizer
KEEP_RATE = 1.0       # keep rate for dropout

def main(_):
  with tf.device('/cpu:0'):
    ### load data from TFRecords file
    index, tfrecords = Reader().index, Reader().read_and_decode('input.tfrecords')
    batches = tf.train.batch(tfrecords, batch_size=BATCH_SIZE, capacity=10)
    data_batch = []
    for i in index:
      data_batch.append(tf.slice(batches[i],[0,1],[-1,1]))
    data_batch = tf.concat(data_batch, 1)
    label_batch = batches['label']

  with tf.device('/gpu:0'):
    ### define neural network form
    w_1 = tf.get_variable('w1',[8, 100])
    b_1 = tf.get_variable('b1',[100])

    w_2 = tf.get_variable('w2',[100,50])
    b_2 = tf.get_variable('b2',[50])

    w_3 = tf.get_variable('w3',[50,50])
    b_3 = tf.get_variable('b3',[50])

    w_out = tf.get_variable('wout',[50, 2])
    b_out = tf.get_variable('bout',[2])

    h_1 = tf.add(tf.matmul(data_batch, w_1), b_1)
    h_1 = tf.nn.relu(h_1)
    h_1 = tf.nn.dropout(h_1, KEEP_RATE)

    h_2 = tf.add(tf.matmul(h_1, w_2), b_2)
    h_2 = tf.nn.relu(h_2)
    h_2 = tf.nn.dropout(h_2, KEEP_RATE)

    h_3 = tf.add(tf.matmul(h_2, w_3), b_3)
    h_3 = tf.nn.relu(h_3)
    h_3 = tf.nn.dropout(h_3, KEEP_RATE)

    out = tf.add(tf.matmul(h_3, w_out), b_out)

    ### define cost function and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=out))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

  with tf.device('/cpu:0'):
    ### define test values
    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(label_batch,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  ### create session and initialize variables
  config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  tf.train.start_queue_runners(sess=sess)

  ### train model
  for step in range(MAX_STEPS):
    _, loss, data_p, acc_p = sess.run([opt, loss_op, data_batch, accuracy])
    if (step+1) % 1000 == 0:
      print("[{:5d}/{:5d}] loss:{:.3f}, train accuracy:{:.3f}".format(step+1, MAX_STEPS, loss, acc_p))
  print("--- Training Finished ---")

  ### test model
#  print("Test  Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:test_x, y:test_y})))

if __name__ == '__main__':
  tf.app.run()
