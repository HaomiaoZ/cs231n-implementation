from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.reshape(x, (x.shape[0], -1)).dot(w)+b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshaped = np.reshape(x, (x.shape[0], -1)) # N*D

    # sum not mean for db
    db = np.sum(dout, axis = 0)
    dw = x_reshaped.T @ dout
    dx = np.reshape(dout @ w.T, x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout
    dx[x<0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # modified non vectorized version

    dx = np.zeros_like(x)
    loss = 0.0

    eps =1e-8

    num_train = x.shape[0]  #N
    num_class = x.shape[1]  #C

    # compute loss and grad wrt score
    # followed https://github.com/mantasu/cs231n/blob/master/assignment2/cs231n/layers.py add numerically stable component by substracting the scores.max
    x = x - x.max(axis=1, keepdims=True)

    for i in range(num_train):
        scores = x[i]
        loss += -np.log(np.exp(scores[y[i]])/(np.sum(np.exp(scores))+eps))
        dx[i] += np.exp(scores)/(np.sum(np.exp(scores)+eps))

        for j in range(num_class):
          if j == y[i]:
              dx[i,j] -= 1.0

    loss /= num_train
    
    dx /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Batch normalization
        sample_mean = np.mean(x, axis = 0) #length D
        sample_var = np.var(x, axis = 0)
        norm_x = (x-sample_mean)/np.sqrt(sample_var+eps)
        y = norm_x*gamma + beta

        # update running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # store variable
        out  = y
        cache = (x, norm_x, sample_mean, sample_var, gamma, beta, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Batch normalization using running mean
        sample_mean = running_mean #length D
        sample_var = running_var
        norm_x = (x-sample_mean)/np.sqrt(sample_var+eps)
        y = norm_x*gamma + beta

        # update running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # store variable
        out  = y
        cache = (x, norm_x, sample_mean, sample_var, gamma, beta, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, norm_x, sample_mean, sample_var, gamma, beta, eps= cache
    N, D = x.shape

    # using derivatives in section 3 in the paper
    dnorm_x = dout*gamma # N*D
    dsample_var = np.sum(((x-sample_mean)*-0.5*np.pow(sample_var+eps,-1.5))*dnorm_x, axis = 0) # elementwise multiplication
    dsample_mean = np.sum(dnorm_x*-1/np.sqrt(sample_var+eps), axis = 0) + dsample_var* np.mean(-2*(x-sample_mean),axis=0)
    dx = dnorm_x*1/np.sqrt(sample_var+eps)+ dsample_var*2*(x-sample_mean)/N+ dsample_mean/N
    dgamma = np.sum(norm_x*dout, axis = 0) # elementwise multiplication
    dbeta = np.sum(dout, axis =0)

    #TODO use computation graph to implement the backward pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # skip alt implementation, use the same implementation as above
    x, norm_x, sample_mean, sample_var, gamma, beta, eps= cache
    N, D = x.shape

    # using derivatives in section 3 in the paper
    dnorm_x = dout*gamma # N*D
    dsample_var = np.sum(((x-sample_mean)*-0.5*np.pow(sample_var+eps,-1.5))*dnorm_x, axis = 0) # elementwise multiplication
    dsample_mean = np.sum(dnorm_x*-1/np.sqrt(sample_var+eps), axis = 0) + dsample_var* np.mean(-2*(x-sample_mean),axis=0)
    dx = dnorm_x*1/np.sqrt(sample_var+eps)+ dsample_var*2*(x-sample_mean)/N+ dsample_mean/N
    dgamma = np.sum(norm_x*dout, axis = 0) # elementwise multiplication
    dbeta = np.sum(dout, axis =0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N,D = x.shape

    # Modified Batch normalization based on slides
    sample_mean = np.mean(x, axis = 1) #length N
    sample_var = np.var(x, axis = 1)
    # a[:, np.newaxis] make row vector to column vector, then broadcast
    norm_x = (x-np.broadcast_to(sample_mean[:,np.newaxis],x.shape))/np.sqrt(np.broadcast_to(sample_var[:,np.newaxis],x.shape)+eps)
    y = norm_x*gamma + beta

    # store variable
    out  = y
    cache = (x, norm_x, sample_mean, sample_var, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x, norm_x, sample_mean, sample_var, gamma, beta, eps= cache

    # modify derivatives in section 3 in the batchnorm paper
    # figure out the implecit broadcasting then everything will be solved (not quite, more changes than needed for backward pass)
    # inspired by https://github.com/mantasu/cs231n/blob/master/assignment2/cs231n/layers.py, transpose dout and work everything with D*N, then transform back
    # if batchnorm implemented properly, can just call batchnorm
    x =x.T
    norm_x =norm_x.T
    dout = dout.T
    gamma = np.broadcast_to(gamma[:,np.newaxis],x.shape)
    
    N, D = x.shape

    dnorm_x = dout*gamma # N*D
    dsample_var = np.sum(((x-sample_mean)*-0.5*np.pow(sample_var+eps,-1.5))*dnorm_x, axis = 0) # elementwise multiplication
    dsample_mean = np.sum(dnorm_x*-1/np.sqrt(sample_var+eps), axis = 0) + dsample_var* np.mean(-2*(x-sample_mean),axis=0)
    dx = dnorm_x*1/np.sqrt(sample_var+eps)+ dsample_var*2*(x-sample_mean)/N+ dsample_mean/N
    dgamma = np.sum(norm_x*dout, axis = 1) # elementwise multiplication (from axis = 0 to axis = 1)
    dbeta = np.sum(dout, axis = 1 )# (from axis = 0 in BN to axis = 1 in LN)

    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.rand(x.shape[0], x.shape[1]) < p # higher the p, morelikely a neuron is kept alive (bool = true)
        out = x*mask/p # elementwise multiplication ,inverted (/p) so test time is untouched

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout*mask/dropout_param["p"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    F, C, HH, WW =w.shape

    pad = conv_param["pad"]
    stride  = conv_param["stride"]

    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        temp_img = x[n] # C*H*W

        # pad img
        temp_pad = np.zeros((C, H, pad))
        temp_img = np.concatenate((temp_pad, temp_img, temp_pad), axis = 2) # pad width (W, left and right)
        
        temp_pad = np.zeros((C, pad, W+2*pad))
        temp_img = np.concatenate((temp_pad, temp_img, temp_pad), axis = 1) # pad height (h, up and bottom)

        for f in range(F):
              
              temp_filter_weight = w[f] # C*HH*WW
              temp_bias = b[f] # dim 1
              
              # calculate each output value
              for i in range(H_out):
                  for j in range(W_out):
                    out[n][f][i][j] = np.sum(temp_img[:, i*stride:i*stride+HH, j*stride:j*stride+WW]*temp_filter_weight) + temp_bias # sum(elementwise multiplication) + bias
                      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param =cache

    N, C, H, W = x.shape
    F, C, HH, WW =w.shape
    N, F, H_out, W_out = dout.shape

    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    
    pad = conv_param["pad"]
    stride  = conv_param["stride"]

    
    # use pad to pervent out of bound
    dx_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    for n in range(N):
        temp_img = x[n] # C*H*W

        # pad img to prevent out of bound calculation
        temp_pad = np.zeros((C, H, pad))
        temp_img = np.concatenate((temp_pad, temp_img, temp_pad), axis = 2) # pad width (W, left and right)
        
        temp_pad = np.zeros((C, pad, W+2*pad))
        temp_img = np.concatenate((temp_pad, temp_img, temp_pad), axis = 1) # pad height (h, up and bottom)

        for f in range(F):
          for i in range(H_out):
              for j in range(W_out):
                  dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += dout[n, f, i, j]*w[f] # constant * (C, HH, WW)
                  dw[f] += dout[n, f, i, j]*temp_img[:,i*stride:i*stride+HH, j*stride:j*stride+WW] #constant * (C, HH, WW) 
                  
    dx =dx_pad[:, :, pad:-pad, pad:-pad] #extract gradient to unpadded input
    db =np.sum(dout, (0, 2, 3)) # (F,), sum over other axes

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W =x.shape

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            
            patch = np.reshape(x[:, :, i*pool_height:i*pool_height+stride, j*pool_width: j*pool_width+stride], (N, C, stride*stride))
            out[:, :, i, j] = np.max (patch, axis = 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x, pool_param = cache
    
    N, C, H, W =x.shape
    N, C, H_out, W_out =dout.shape

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    # gradient router, need to find the location of the max

    dx = np.zeros(x.shape)

    for i in range(H_out):
        for j in range(W_out):
            # assume reshape use the same order back and forth
            patch = np.reshape(x[:, :, i*pool_height:i*pool_height+stride, j*pool_width: j*pool_width+stride], (N, C, stride*stride))
            ind = np.argmax(patch, axis = 2) # find indices
            
            eye_mat = np.eye(stride*stride)
            for n in range(N):
                for c in range(C):
                    one_hot_vec = eye_mat[ind[n,c]]
                    mask =np.reshape(one_hot_vec,(stride, stride))
                    dx[n,c,i*pool_height:i*pool_height+stride, j*pool_width: j*pool_width+stride] =mask*dout[n,c,i,j]
                    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # x(N,C,H,W) -> (N,D), gamma (C,)->(D, ), beta (C,)->(D,) D = C*H*W
    N,C,H,W = x.shape
    gamma = np.broadcast_to(gamma.reshape((C,1)),(C,H*W))
    gamma =gamma.reshape(-1)
    beta = np.broadcast_to(beta.reshape((C,1)),(C,H*W))
    beta = beta.reshape(-1)
    out, cache = batchnorm_forward(x.reshape((N,-1)), gamma, beta, bn_param)
    out = out.reshape(x.shape)
    #(N,D)->(N,C,H,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dx, dgamma, dbeta = batchnorm_backward(dout.reshape((N,-1)), cache) # N,C,H,W -> N,D
    dx = dx.reshape(dout.shape) #N,D -> N,C,H,W
    dgamma = np.sum(dgamma.reshape((C,-1)), axis = 1) #D,-> C,
    dbeta = np.sum(dbeta.reshape((C,-1)), axis = 1) #D,-> C,

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = np.zeros(x.shape)
    cache = []
    N,C,H,W = x.shape
    # per group, it is layernorm
    for i in range (G):
        x_temp = x[:,i*C//G:(i+1)*C//G, :, :].reshape((N,-1)) # extract (N, C//G, H, W) -> (N, D)

        gamma_temp = gamma[:, i*C//G:(i+1)*C//G,:,:] # extract (1, C//G, 1, 1) 
        gamma_temp = np.broadcast_to(gamma_temp,(1,C//G,H,W)).reshape(-1) # broadcast to (1, C//G, H, W)->(D,)

        beta_temp = beta[:, i*C//G:(i+1)*C//G,:,:] # extract (1, C//G, 1, 1) 
        beta_temp = np.broadcast_to(beta_temp,(1,C//G,H,W)).reshape(-1) # broadcast to (1, C//G, H, W)->(D,)

        out_temp, cache_temp = layernorm_forward(x_temp, gamma_temp, beta_temp, gn_param)

        out_temp = np.reshape(out_temp,(N,C//G, H, W)) # (N, D) -> (N, C//G, H, W) 
        
        out[:,i*C//G:(i+1)*C//G, :, :] = out_temp
        cache.append(cache_temp)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    G = len(cache)
    N, C, H, W = dout.shape
    dx = np.zeros(dout.shape)
    dgamma = np.zeros((1, C, 1, 1))
    dbeta = np.zeros((1, C, 1, 1))

    for i in range(G):
        cache_temp = cache[i]
        dout_temp = dout[:,i*C//G:(i+1)*C//G, :, :].reshape((N,-1)) # extract (N, C//G, H, W) -> (N, D)
        dx_temp, dgamma_temp, dbeta_temp = layernorm_backward(dout_temp, cache_temp)

        dx[:,i*C//G:(i+1)*C//G, :, :] = dx_temp.reshape((N, C//G, H, W)) # (N,D) -> (N, C//G, H, W)
        dgamma[:,i*C//G:(i+1)*C//G, :, :] = np.sum(dgamma_temp.reshape((C//G,-1)), axis=1).reshape((C//G,1,1)) # reshape to C//G, H*W, sum up, then reshape from (C//G,) to(C//G, 1, 1)
        dbeta[:,i*C//G:(i+1)*C//G, :, :] = np.sum(dbeta_temp.reshape((C//G,-1)), axis=1).reshape((C//G,1,1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
