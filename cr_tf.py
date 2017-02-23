from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reader import Reader

MAX_STEPS = 3000
BATCH_SIZE = 10
LEARNING_RATE = 0.005

def main(_):
  ### load data from TFRecords file
  indices, tfrecords = Reader().index, Reader().read_and_decode('input.tfrecords')
  batches = tf.train.batch(tfrecords, batch_size=BATCH_SIZE, capacity=200)
  data_batch = []
  for index in indices:
    data_batch.append(tf.slice(batches[index],[0,1],[-1,1]))
  data_batch = tf.concat(data_batch, 1)
  label_batch = batches['label']
  print(data_batch, label_batch)

  ### define neural network form
  w_1 = tf.get_variable('w1',[8, 100])
  b_1 = tf.get_variable('b1',[100])

  w_2 = tf.get_variable('w2',[100,50])
  b_2 = tf.get_variable('b2',[50])

  w_out = tf.get_variable('wout',[50, 2])
  b_out = tf.get_variable('bout',[2])

  h_1 = tf.add(tf.matmul(data_batch, w_1), b_1)
  h_1 = tf.nn.relu(h_1)
  h_1 = tf.nn.dropout(h_1, 0.60)

  h_2 = tf.add(tf.matmul(h_1, w_2), b_2)
  h_2 = tf.nn.relu(h_2)
  h_2 = tf.nn.dropout(h_2, 0.60)

  out = tf.add(tf.matmul(h_2, w_out), b_out)

  ### define cost function and optimizer
  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=out))
  opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

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
    _, loss, data_p = sess.run([opt, loss_op, data_batch])
    if (step+1) % 100 == 0:
      print(data_p)
      print("[{}/{}] loss:{:.3f}".format(step+1, MAX_STEPS, loss))
  print("--- Training Finished ---")

  ### test model
#  print("Train Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:data_batch, y:label_batch})))
#  print("Test  Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:test_x, y:test_y})))

if __name__ == '__main__':
  with tf.device('/cpu:0'):
    tf.app.run()
