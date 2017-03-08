from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Decorator for lazy loading of tensorflow graph
# refer https://danijar.github.io/structuring-your-tensorflow-models

import functools
import tensorflow as tf

def doublewrap(function):
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
      return function(args[0])
    else:
      return lambda wrapee: function(wrapee, *args, **kwargs)
  return decorator

@doublewrap
def lazyload(function, scope=None, *args, **kwargs):
  attribute = '_cache_' + function.__name__
  name = scope or function.__name__
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(name, *args, **kwargs):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator

