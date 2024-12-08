from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for l in range (self.num_layers):
          if l == 0:
            self.params[f'W{l + 1}'] = np.random.normal(0.0, weight_scale, size=(input_dim, hidden_dims[l]))
            self.params[f'b{l + 1}'] = np.zeros(hidden_dims[l])
          elif l == self.num_layers - 1:
            self.params[f'W{l + 1}'] = np.random.normal(0.0, weight_scale, size=(hidden_dims[l-1], num_classes))
            self.params[f'b{l + 1}'] = np.zeros(num_classes)
          else:   
            self.params[f'W{l + 1}'] = np.random.normal(0.0, weight_scale, size=(hidden_dims[l-1], hidden_dims[l]))
            self.params[f'b{l + 1}'] = np.zeros(hidden_dims[l])

          if self.normalization == "batchnorm" and l < self.num_layers - 1:
              self.params[f'gamma{l + 1}'] = np.ones(hidden_dims[l])
              self.params[f'beta{l + 1}'] = np.zeros(hidden_dims[l])

          if self.normalization == "layernorm" and l < self.num_layers - 1:
              self.params[f'gamma{l + 1}'] = np.ones(hidden_dims[l])
              self.params[f'beta{l + 1}'] = np.zeros(hidden_dims[l])    

            
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        self.ln_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
      
        """Compute loss and gradient for the fully connected net.
  
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        cache = {}

        if self.normalization == "batchnorm":
          for l in range (self.num_layers):
            W = self.params[f'W{l + 1}']
            b = self.params[f'b{l + 1}']

            if l == 0:
              gamma = self.params[f'gamma{l + 1}']
              beta = self.params[f'beta{l + 1}']
              output, cache[l] = general_relu_forward(X, W, b,  gamma, beta, self.bn_params[l])

            if l != 0 and l < self.num_layers - 1:
              gamma = self.params[f'gamma{l + 1}']
              beta = self.params[f'beta{l + 1}']
              output, cache[l] = general_relu_forward(output, W, b, gamma, beta, self.bn_params[l])

            if l == self.num_layers - 1:
              # Last layer
              output, cache_affine = affine_forward(output, W, b)
              cache[l] = (cache_affine, ) # No ReLU in the last layer

          scores = output

        elif self.normalization == "layernorm":
          for l in range (self.num_layers):
            W = self.params[f'W{l + 1}']
            b = self.params[f'b{l + 1}']

            if l == 0:
              gamma = self.params[f'gamma{l + 1}']
              beta = self.params[f'beta{l + 1}']
              output, cache[l] = laylernorm_relu_forward(X, W, b,  gamma, beta, self.ln_params[l])

            if l != 0 and l < self.num_layers - 1:
              gamma = self.params[f'gamma{l + 1}']
              beta = self.params[f'beta{l + 1}']
              output, cache[l] = laylernorm_relu_forward(output, W, b, gamma, beta, self.ln_params[l])

            if l == self.num_layers - 1:
              # Last layer
              output, cache_affine = affine_forward(output, W, b)
              cache[l] = (cache_affine, ) # No ReLU in the last layer

          scores = output

        elif self.use_dropout:
          for l in range(self.num_layers):
            W = self.params[f'W{l + 1}']
            b = self.params[f'b{l + 1}']
            if l == 0:
              out, cache[l] = affine_relu_dropout_forward(X, W, b, self.dropout_param)
            elif l == self.num_layers - 1:
              out, cache_affine = affine_forward(out, W, b)
              cache[l] = (cache_affine, ) # 

            else:
              out, cache[l] = affine_relu_dropout_forward(out, W, b, self.dropout_param)

          scores = out  

        else: 
          for l in range(self.num_layers):
            W = self.params[f'W{l + 1}']
            b = self.params[f'b{l + 1}']
            if l == 0:
              # First layer
              out, cache_affine = affine_forward(X, W, b)
              out, cache_relu = relu_forward(out)
              cache[l] = (cache_affine, cache_relu)
            elif l == self.num_layers - 1:
              # Last layer
              out, cache_affine = affine_forward(out, W, b)
              cache[l] = (cache_affine, ) # No ReLU in the last layer
            else:
              out, cache_affine = affine_forward(out, W, b)
              out, cache_relu = relu_forward(out)
              cache[l] = (cache_affine, cache_relu)
          scores = out
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dout = softmax_loss(scores, y) # Final softmax Loss
        for l in range(self.num_layers):
          W = self.params[f'W{l + 1}']
          loss += 0.5 * self.reg * np.sum(W ** 2) # Regularization from all W's
        
        if self.normalization == "batchnorm":
      
          for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
              cache_affine = cache[l][0]
              dout, dw, db = affine_backward(dout, cache_affine)

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

            else:
              fc_cache, bn_cache, relu_cache = cache[l]
              dout, dw, db, dgamma, dbeta = general_relu_backward(dout, (fc_cache, bn_cache, relu_cache))

              # gradients for affine
              W = fc_cache[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

              # gradients for batch norm
              grads[f'gamma{l + 1}'] = dgamma
              grads[f'beta{l + 1}'] = dbeta

        elif self.normalization == "layernorm":
      
          for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
              cache_affine = cache[l][0]
              dout, dw, db = affine_backward(dout, cache_affine)

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

            else:
              fc_cache, ln_cache, relu_cache = cache[l]
              dout, dw, db, dgamma, dbeta = laylernorm_relu_backward(dout, (fc_cache, ln_cache, relu_cache))

              # gradients for affine
              W = fc_cache[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

              # gradients for batch norm
              grads[f'gamma{l + 1}'] = dgamma
              grads[f'beta{l + 1}'] = dbeta

        elif self.use_dropout:
          for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
              cache_affine = cache[l][0]
              dout, dw, db = affine_backward(dout, cache_affine)

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

            else:
              cache_affine, cache_relu, cache_drop = cache[l]
              dout, dw, db = affine_relu_dropout_backward(dout, (cache_affine, cache_relu, cache_drop))

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

        else:

          for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
              cache_affine = cache[l][0]
              dout, dw, db = affine_backward(dout, cache_affine)

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

            else:
              cache_affine, cache_relu = cache[l]
              dout = relu_backward(dout, cache_relu)
              dout, dw, db = affine_backward(dout, cache_affine)

              W = cache_affine[1]
              dw += self.reg * W

              grads[f'W{l + 1}'] = dw
              grads[f'b{l + 1}'] = db

            pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
