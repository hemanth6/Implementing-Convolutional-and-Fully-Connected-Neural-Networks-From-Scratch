from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        #pass
        for n,v in layer.params.items():
            if n not in self.velocity.keys():
                self.velocity[n] = np.zeros(layer.params[n].shape)
            self.velocity[n] = (self.momentum*self.velocity[n]) - (self.lr*layer.grads[n])
            layer.params[n] = layer.params[n] + self.velocity[n]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        #pass
        for n,v in layer.params.items():
            if n not in self.cache.keys():
                self.cache[n] = np.zeros(layer.params[n].shape)
            self.cache[n] = (self.decay*self.cache[n]) + (1.0-self.decay) * np.square(layer.grads[n])
            layer.params[n] -= self.lr * layer.grads[n] / np.sqrt(self.cache[n] + self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        #pass
        for n,v in layer.params.items():
            if n not in self.vt.keys():
                self.vt[n] = np.zeros(layer.params[n].shape)
            if n not in self.mt.keys():
                self.mt[n] = np.zeros(layer.params[n].shape)
            self.t += 1
            self.mt[n] = self.beta1 * self.mt[n] + (1.0 - self.beta1) * layer.grads[n]
            self.vt[n] = self.beta2 * self.vt[n] + (1.0 - self.beta2) * np.square(layer.grads[n])
            mt_temp = self.mt[n] / (1.0 - self.beta1**(self.t))
            vt_temp = self.vt[n] / (1.0 - self.beta2**(self.t))
            layer.params[n] -= self.lr * mt_temp/ (np.sqrt(vt_temp) + self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
