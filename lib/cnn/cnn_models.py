from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 3, 1, 0, name="conv2d"),
            MaxPoolingLayer(2, 2, name="maxp"),
            flatten(name="flat"),
            fc(27, 5, 0.02, name="fc_test1")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            
            
            ConvLayer2D(3, 3, 8, 1, 0, name="conv2d1"),
            leaky_relu(name="relu1"),
            ConvLayer2D(8, 4, 8, 1, 0, name="conv2d2"),
            leaky_relu(name="relu2"),
            MaxPoolingLayer(3, 3, name="maxp1"),
            dropout(0.8, name="drop1"),
            flatten(name="flat2"),
            fc(648, 100, 0.02, name="fc_test1"),
            leaky_relu(name="relu3"),
            dropout(0.8, name="drop2"),
            fc(100, 10, 0.02, name="fc_test2")
            
            ########### END ###########
        )
