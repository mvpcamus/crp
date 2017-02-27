from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reader import Reader

class FullConnected(object):
  '''
  Restricted Boltzmann Machine
  Args:
    input: input tensor array = x
    shape: [input size, output size]
  Return:
    output nodes tensor array = w * x + b
  '''
  def __init__(self, x, shape):
    self.x = x
    self.w = tf.get_variable('w', shape, trainable=True)
    self.b = tf.get_variable('b', shape[1], trainable=True)
  def output(self):
    return tf.add(tf.matmul(self.x, self.w), self.b)

class LSTM(object):
  def __init__(self,input, shape):
    pass

class BNormal(object):
  def __init__(self, x, shape, train):
    self.x = x
    self.train = train
    self.beta = tf.Variable(tf.constant(0.0, shape=shape), name='beta', trainable=True)
    self.gamma = tf.Variable(tf.constant(1.0, shape=shape), name='gamma', trainable=True)
  def output(self):
    batch_mean, batch_var = tf.nn.moments(self.x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(self.train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(self.x, mean, var, self.beta, self.gamma, 1e-3)
 
def batch_normalize(x, shape, train_phase):
  with tf.variable_scope('bn'):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[shape]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
  return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

def main(_):
  MAX_STEPS = 20000
  BATCH_SIZE = 30
  LEARNING_RATE = 0.005 # learning rate for optimizer
  KEEP_RATE = 1.0       # keep rate for dropout
  
  train_phase = tf.placeholder(tf.bool, name='phase_train')
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
    with tf.variable_scope('hidden1'):
      h1 = FullConnected(data_batch, [8, 100]).output()
      h1_br = BNormal(h1, [100], train_phase).output()
      h1_relu = tf.nn.relu(h1_br)
      h1_out = tf.nn.dropout(h1_relu, KEEP_RATE)

    with tf.variable_scope('hidden2'):
      h2 = FullConnected(h1_out, [100, 50]).output()
      h2_br = BNormal(h2, [50], train_phase).output()
      h2_relu = tf.nn.relu(h2)
      h2_out = tf.nn.dropout(h2_relu, KEEP_RATE)

    with tf.variable_scope('hidden3'):
      h3 = FullConnected(h2_out, [50, 50]).output()
      b3_br = BNormal(h3, [50], train_phase).output()
      h3_relu = tf.nn.relu(h3)
      h3_out = tf.nn.dropout(h3_relu, KEEP_RATE)

    with tf.variable_scope('output'):
      out = FullConnected(h3_out, [50, 2]).output()

    ### define cost function and optimizer
  with tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
  with tf.device('/gpu:0'):
    decayed_lr = tf.train.exponential_decay(LEARNING_RATE, global_step, int(MAX_STEPS/3), 0.95, staircase=True)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_batch, logits=out) 
    loss_op = tf.reduce_mean(entropy)
    opt = tf.train.AdamOptimizer(decayed_lr).minimize(loss_op, global_step=global_step)

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
    _, loss, data_p = sess.run([opt, loss_op, data_batch], {train_phase:True})
    if (step+1) % 1000 == 0:
      acc_p = accuracy.eval({train_phase:False}, session=sess)
      print("[{:5d}/{:5d}] loss:{:.3f}, train accuracy:{:.3f}".format(step+1, MAX_STEPS, loss, acc_p))
  print("--- Training Finished ---")

  ### test model
#  print("Test  Accuracy: {:.3f}".format(sess.run(accuracy, feed_dict={x:test_x, y:test_y})))

if __name__ == '__main__':
  tf.app.run()
