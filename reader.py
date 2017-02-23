from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Reader(object):
  def __init__(self):
    # currency indices of input records
    self.index = ['USD', 'JPY', 'CNY', 'EUR', 'GBP', 'CHF', 'CAD', 'AUD']

  def read_and_decode(self, filename):
    '''
    Args:
      filename = input TFRecords file path
    Return:
      dataDict = decoded data in dictionary type
    '''
    reader = tf.TFRecordReader()
    _, example = reader.read(tf.train.string_input_producer([filename], num_epochs=None))
    features = {'label': tf.FixedLenFeature([2], tf.int64)}
    for index in self.index:
      features[index] = tf.FixedLenFeature([2], tf.float32)
    dataDict = tf.parse_single_example(example, features=features)
    return dataDict

  def print_data(self, filename):
    # print title of currency indices
    text = '   '
    for i in self.index: text += (i + '     ')
    text += 'label'
    print(text)
    # print data values
    example = tf.train.Example()
    for serialized in tf.python_io.tf_record_iterator(filename):
      example.ParseFromString(serialized)
      text = ''
      for index in self.index:
        text += '{:7.2f} '.format(example.features.feature[index].float_list.value[0])
      text += '{:6d}'.format(example.features.feature['label'].int64_list.value[0])
      print(text)
    return

if __name__ == '__main__':
  Reader().print_data('input.tfrecords')
