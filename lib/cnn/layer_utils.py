from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        #pass
        output_shape[0] = input_size[0]
        output_shape[1] = int(((input_size[1] - self.kernel_size + (2 * self.padding)) / self.stride) + 1)
        output_shape[2] = int(((input_size[2] - self.kernel_size + (2 * self.padding)) / self.stride) + 1)
        output_shape[3] = self.number_filters
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        #pad the input image according to self.padding (see np.pad)
        #pass
        h_f, w_f, _, n_f = self.params[self.w_name].shape
        
        #output_p = np.pad(img, self.padding, 'constant')
        output_p = np.pad(img, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)),'constant')
        #output_p = np.pad(img, ((int(self.padding), int(self.padding)), (int(self.padding), int(self.padding)),'constant')
        
        #iterate over output dimensions, moving by self.stride to create the output
        #pass
        #print(output_shape)
        output = np.zeros(output_shape)
        

        
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                
                output[:, i, j, :] = np.sum(
                    ((output_p[:, h_start:h_end, w_start:w_end, :, np.newaxis]) *
                    (self.params[self.w_name][np.newaxis, :, :, :, :])),
                    axis=(1, 2, 3)
                )
        output = output + self.params[self.b_name]
        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        #pass
        _, h_out, w_out, _ = dprev.shape
        n, h_in, w_in, _ = img.shape
        h_f, w_f, _, _ = self.params[self.w_name].shape
        
        prev_pad = np.pad(img, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)),'constant')
        output = np.zeros_like(prev_pad)
        self.grads[self.b_name] = dprev.sum(axis=(0, 1, 2)) 
        self.grads[self.w_name] = np.zeros_like(self.params[self.w_name])
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.params[self.w_name][np.newaxis, :, :, :, :] *
                    dprev[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self.grads[self.w_name] += np.sum(
                    prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    dprev[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )
        dimg = output[:, self.padding:self.padding+h_in, self.padding:self.padding+w_in, :]
        
                
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        #pass
        temp = np.array(img, copy=True)
        n, h_in, w_in, c = img.shape
        h_out = 1 + (h_in - self.pool_size) // self.stride
        w_out = 1 + (w_in - self.pool_size) // self.stride
        output = np.zeros((n, h_out, w_out, c))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                img_slice = img[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(img_slice, axis=(1, 2))
                
        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        #pass
        #n, c, h, w = img.shape
        dn, _, _, dc = dprev.shape

        
        for d in range(dc):
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * self.stride
                    h_end = h_start + h_pool
                    w_start = j * self.stride
                    w_end = w_start + w_pool
                    temp = img[:, h_start:h_end, w_start:w_end, d]
                    mask = (temp == np.max(temp, axis=(1,2), keepdims=True))
                    dimg[:, h_start:h_end, w_start:w_end, d] += dprev[:, i, j, d][:, np.newaxis, np.newaxis] * mask
        
                
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
