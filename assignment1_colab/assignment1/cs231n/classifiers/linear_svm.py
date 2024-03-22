from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) 
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range (num_train):
        f = X[i].dot(W)
        for j in range (num_classes):
            if j != y[i]:
                margin = f[j] - f[y[i]] +1
                if margin > 0 :
                    loss += margin
                    dW[:,j] += X[i]
                    dW[:,y[i]] -= X[i]
        
    loss = loss/num_train
    
    loss += reg * np.sum(W*W)
    
    dW += 2*reg*W
    
    return loss, dW 

def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)
    
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    scores = X.dot(W)
    correct_class_score = scores[range(num_train), list(y)].reshape(-1,1)
    margin = np.maximum(0, scores - correct_class_score+1)
    margin[range(num_train), list(y)] = 0
    
    loss = np.sum(margin) / num_train + 0.5*reg*np.sum(W*W)
    
    mat_margin = np.zeros((num_train, num_class))
    
    # print(margin.shape)
    # print(mat_margin.shape)
    mat_margin[margin > 0] = 1
    mat_margin[range(num_train), list(y)] = 0
    mat_margin[range(num_train), list(y)] -=np.sum(mat_margin, axis = 1)
    
    dW = reg *W + (X.T).dot(mat_margin)/num_train
    
    return loss, dW