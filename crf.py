import numpy as np


class CRF(object):

    def __init__(self,
                input_size,
                classes,
                learning_rate=0.001):
        self.input_size = input_size
        self.classes = classes
        self.learning_rate = learning_rate
        self.ready()
 
    def ready(self):
        self.w = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]
        self.b = np.zeros((self.classes))

        self.w_edge = np.zeros((self.classes, self.classes))

    def forward(self, input, target):
        """
        input, (seq_length, input_size)
        target, (seq_length)
        """
        # convert label to one-hot vector
        self.target_onehot = np.zeros((self.seq_length, self.classes))
        self.target_onehot[np.arange(self.seq_length), target] = 1

        # propagate belief using forward-backward algorithm
        self.belief_propagation(input)

        # compute 
        s = 0
        s = s + np.matmul(self.target_onehot[0], np.matmul(input[0], self.w[0]) + np.matmul(input[1], self.w[2]) + self.b)
        for i in range(1, self.seq_length - 1):
            s = s + np.matmul(self.target_onehot[i], np.matmul(input[i], self.w[0]) + np.matmul(input[i-1], self.w[1]) + np.matmul(input[i+1], self.w[2]) + self.b) +\
                    np.matmul(np.matmul(self.target_onehot[i-1], self.w_edge), self.target_onehot[i])
        s = s + np.matmul(self.target_onehot[self.seq_length - 1], np.matmul(input[self.seq_length - 1], self.w[0]) + np.matmul(input[self.seq_length - 2], self.w[1]) + self.b) + np.matmul(np.matmul(self.target_onehot[self.seq_length - 2], self.w_edge), self.target_onehot[self.seq_length - 1])

        # compute loss
        self.logZ = np.log(self.Z)
        loss = self.logZ - s

        return loss / self.seq_length

    def belief_propagation(self, input):
        # init parameters
        self.seq_length = input.shape[0]
        self.alpha = np.zeros((self.classes, self.seq_length))
        self.beta = np.zeros((self.classes, self.seq_length))
        self.omega = np.zeros((self.classes, self.classes, self.seq_length))

        # compute clique potential omega
        for s1 in range(self.classes):
            for s2 in range(self.classes):
                for j in range(1, self.seq_length - 1):
                    xw = np.matmul(input[j], self.w[0]) + np.matmul(input[j-1], self.w[1]) + np.matmul(input[j+1], self.w[2]) + self.b
                    self.omega[s1, s2, j] = np.exp(self.w_edge[s1, s2] + xw[s2])

                xw = np.matmul(input[self.seq_length-1], self.w[0]) + np.matmul(input[self.seq_length-2], self.w[1]) + self.b
                self.omega[s1, s2, self.seq_length - 1] = np.exp(self.w_edge[s1, s2] + xw[s2])

        # compute forward terms
        self.alpha[:, 0] = np.exp(np.matmul(input[0], self.w[0]) + np.matmul(input[1], self.w[2]) + self.b)
        for i in range(1, self.seq_length):
            self.alpha[:, i] = np.matmul(self.alpha[:, i-1], self.omega[:, :, i])

        # compute back ward terms
        self.beta[:, self.seq_length-1] = 1
        for i in range(self.seq_length-2, -1, -1):
            self.beta[:, i] = np.matmul(self.omega[:, :, i+1], self.beta[:, i+1])

        # compute partition function
        self.Z = np.sum(self.alpha[:, -1])

    def backward(self, input, target):
        self.dw = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]
        self.db = np.zeros((self.classes))
        self.dw_edge = np.zeros((self.classes, self.classes))

        marginal_prob = np.zeros((self.seq_length, self.classes, self.classes))
        for i in range(self.seq_length-1):
            for s1 in range(self.classes):
                for s2 in range(self.classes):
                    marginal_prob[i, s1, s2] = self.alpha[s1, i] * self.omega[s1, s2, i+1] * self.beta[s2, i+1]
        marginal_prob = marginal_prob / self.Z
        marginal_uni_prob = self.alpha*self.beta / self.Z

        for i in range(self.seq_length):
            self.dw[0] -= np.matmul(input[i, None].T, self.target_onehot[i, None] - marginal_uni_prob[:, i][None, :])

        for i in range(self.seq_length-1):
            self.dw[1] -= np.matmul(input[i, None].T, self.target_onehot[i + 1, None] - marginal_uni_prob[:, i+1][None, :])

        for i in range(1, self.seq_length):
            self.dw[2] -= np.matmul(input[i, None].T, self.target_onehot[i - 1, None] - marginal_uni_prob[:, i-1][None, :])

        for i in range(self.seq_length):
            self.db -=  self.target_onehot[i] - marginal_uni_prob[:, i]

        for i in range(self.seq_length-1):
            f = np.zeros((self.classes, self.classes))
            f[target[i], target[i+1]] = 1
            self.dw_edge -= f - marginal_prob[i]

        return self.dw, self.dw_edge, self.db

    def update(self):
        self.w[0] = self.w[0] - self.learning_rate*self.dw[0]
        self.w[1] = self.w[1] - self.learning_rate*self.dw[1]
        self.w[2] = self.w[2] - self.learning_rate*self.dw[2]
        self.w_edge = self.w_edge - self.learning_rate*self.dw_edge
        self.b = self.b - self.learning_rate*self.db

    def predict(self, input):
        self.belief_propagation(input)
        marginal_uni_prob = self.alpha * self.beta / self.Z
        pred = np.argmax(marginal_uni_prob, axis=0)
        return pred
