from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import argparse
import numpy as np
import tensorflow as tf

class CSVParse(object):

  def __init__(self):
    self.delimiter = ';'   # delimiter of csv source to correct
    self.dialect = 'excel' # dialiect of csv source to correct
    self.point = ','       # decimal point type in csv source to correct
    self.index = True      # input csv source contains index row

  def convert(self, infile, outfile, target):
    '''
    Convert original csv file to TFRecords file
    args:
      infile = string of input CSV filepath
      outfile = string of output tfrecords filepath
      target = target currency for label [USD, JPY, CNY, EUR, GBP, CHF, CAD, AUD]
    '''
    data = [] 
    with open(infile) as csvfile:
      reader = csv.reader(csvfile, delimiter='\t')
      if self.index == True:
        index = reader.next()
      else:
        index = ['USD', 'JPY', 'CNY', 'EUR', 'GBP', 'CHF', 'CAD', 'AUD']
      for row in reader:
        values = []
        for col in row:
          try:
            values.append(float(col))
          except:
            print('ERROR: inadequate data value: ', col)
            return
        data.append(values)
    if data == []: return
    else: data = np.array(data)

    # calculate currency variates from the previous days
    diff = []
    for i in xrange(len(data)-1):
      diff.append(data[i+1] - data[i])

    # labels to target currency: [1,0] = will increase, [0,1] = will decrease
    target = target
    label = []
    for i in xrange(len(diff)-1):
      x = [1,0] if (diff[i+1][index.index(target)]>0) else [0,1]
      label.append(x)

    writer = tf.python_io.TFRecordWriter(outfile)
    for i in range(len(label)):
      feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label[i]))}
      for j,key in enumerate(index):
        feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[data[i+1][j],diff[i][j]]))
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
    writer.close()

    print ('Successfully created TFRecord File [{}]'.format(outfile))
    print ('Data source: [{}], Target Currency: [{}]'.format(infile, target))
    print ('Inputs: {}'.format(index))
    return

  def correct(self, infile, outfile):
    '''
    Convert excel generated csv file to original csv
    args:
      infile = string of input CSV filepath
      outfile = string of output CSV filepath
    '''
    data = []
    with open(infile) as csvfile:
      reader = csv.reader(csvfile, dialect=self.dialect, delimiter=self.delimiter)
      if self.index == True:
        data.append(reader.next())
      for row in reader:
        values = []
        for col in row:
          try:
            values.append(float(col.replace(self.point, '.')))
          except:
            if not col == '':
              print('ERROR: inadequate data value detected: ', col)
              return
            else:
              values = []
              break
        if not values ==[]: data.append(values)
    with open(outfile, 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter='\t')
      for row in data:
        writer.writerow(row)
      print('Corrected file is saved as [{}]'.format(outfile))
    return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=str, default='input.csv', help='Input CSV Filename')
  parser.add_argument('-o', type=str, default='input.tfrecords', help='Output TFRecord Filename')
  parser.add_argument('-t', type=str, default='EUR', help='Target Currency, ex) USD, EUR')
  parser.add_argument('-c', type=str, default='', help='Correct CSV file format before parsing')
  FLAGS = parser.parse_args()
  csvparse = CSVParse()
  if not FLAGS.c == '': csvparse.correct(FLAGS.c, FLAGS.i)
  csvparse.convert(FLAGS.i, FLAGS.o, FLAGS.t)
