import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.fcl1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.fcl2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu = ReLULayer()
        self.reg = reg
        self.w1 = self.fcl1.params()['W']
        self.w2 = self.fcl2.params()['W']
        self.b1 = self.fcl1.params()['B']
        self.b2 = self.fcl1.params()['B']
        # TODO Create necessary layers
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        [self.params()[param].grad.fill(0) for param in self.params().keys()]
        # raise Exception("Not implemented!")

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        hidden_res_forward = self.fcl1.forward(X)
        hidden_res_forward = self.relu.forward(hidden_res_forward)
        output = self.fcl2.forward(hidden_res_forward)
        loss, dprediction = softmax_with_cross_entropy(output.T, y)
        hidden_res_backward = self.fcl2.backward(dprediction.T)
        hidden_res_backward = self.relu.backward(hidden_res_backward)
        self.fcl1.backward(hidden_res_backward)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad   # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        pred = np.argmax(softmax(self.fcl2.forward(self.relu.forward(self.fcl1.forward(X)))), 1)
        display(pred)
        # raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {"W1": self.w1, "W2": self.w2, "B1": self.b1, "B2": self.b2}

        # TODO Implement aggregating all of the params

        # raise Exception("Not implemented!")

        return result
