from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    # print("num_clc", num_class)
    
    score = X.dot(W)

    for i in range (num_train):
        loss_i = - score[i][y[i]] + np.log(np.sum(np.exp(score[i])))
        loss += loss_i
        
        for j in range (num_class):
            # print(j)
            u = np.exp(score[i][j]) / np.sum(np.exp(score[i]))
            
            if j == y[i]:
                dW[:, j] += (-1 + u) * X[i]
            else:
                dW[:, j] = u * X[i]
        
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    
    dW = dW / num_train + reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    
    score = X.dot(W)
    score_exp = np.exp(score)
    score_sf = score_exp / np.sum(score_exp, axis = 1).reshape(-1, 1)
    
    loss = -np.sum(np.log(score_sf[range(num_train), list(y)]))
    
    loss = loss/num_train + 0.5 * reg * np.sum(W*W)
    
    dS  = score_sf.copy()
    dS[range(num_train), list(y)] -= 1
    
    dW = (X.T).dot(dS)
    dW = dW/num_train + 0.5*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

