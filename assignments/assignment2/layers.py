import numpy as np
import math


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    # raise Exception("Not implemented!")
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * W * reg_strength
    return loss, grad


def softmax(predictions):
    '''
        Computes probabilities from scores

        Arguments:
          predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output

        Returns:
          probs, np array of the same shape as predictions -
            probability for every class, 0..1
        '''
    # TODO implement softmax
    if len(predictions.shape) == 1:
        predictions -= np.max(predictions)
    else:
        predictions -= np.max(predictions, 0)
    probs = np.exp(predictions) / np.sum(np.exp(predictions), 0)
    return probs
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropys
    if isinstance(target_index, int):
        return -math.log(probs[target_index])
    loss = np.zeros(probs[0].shape)
    if len(target_index.shape) == 1:
        for col in range(probs.shape[1]):
            loss[col] = -math.log(probs[target_index[col]][col])
    else:
        for col in range(probs.shape[1]):
            loss[col] = -math.log(probs[target_index[col][0]][col])
    return np.mean(loss)
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    dprediction = softmax(np.copy(predictions))
    loss = cross_entropy_loss(dprediction, target_index)
    if isinstance(target_index, int):
        dprediction[target_index] -= 1
    else:
        for col in range(target_index.shape[0]):
            dprediction[target_index[col], col] -= 1
        dprediction = dprediction / target_index.shape[0]
    return loss, dprediction
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.x = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.x = X
        return np.maximum(X, 0)
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # display("backward before")
        relu_grad = self.x > 0
        return d_out * relu_grad
        raise Exception("Not implemented!")

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = Param(X)
        return np.dot(self.X.value, self.W.value) + self.B.value
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradients
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        d_result = np.dot(d_out, self.W.value.T)
        d_w = np.dot(self.X.value.T, d_out)
        d_b = np.array([np.sum(d_out, axis=0)])
        self.W.grad += d_w
        self.B.grad += d_b
        return d_result
        raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}
