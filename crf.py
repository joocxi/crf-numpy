import numpy as np


class CRF(Object):

    def __init__(self,
                input
                classes):
        self.input = input
        self.classes = classes
 
    def ready(self):
        self.alpha = np.zeros((0, 0))
        self.beta = np.zeros((0, 0))
        self.w = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]
        self.b = np.zeros((self.classes))

        self.w_edge = np.zeros((self.n_classes, self.n_classes))

        self.dw = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]

        self.db = [np.zeros((self.classes))]
        self.dw_edge = np.zeros((self.n_classes, self.n_classes))

    def forward(self, input, target):
        """
        input, (seq_length, input_size)
        target, (seq_length)
        """
        self.seq_length = input.shape[0]
        target_onehot = np.zeros((self.seq_length, self.n_classes))
        target_onehot[np.arange(self.seq_length), target) = 1

        self.alpha = np.zeros((self.seq_length, self.n_classes))
        self.beta = np.zeros((self.seq_length, self.n_classes))

        # compute unary log factors -log Z_d
        s = 0
        s = s + np.matmul(target_onehot[0], np.matmul(input[0], self.w[0]) + np.matmul(input[1], self.w[2]) + self.b) + np.matmul(np.matmul(target_onehot[0], self.w_edge), target_onehot[1])

        for i in range(1, seq_length - 1):
            s = s + np.matmul(target_onehot[i], np.matmul(input[i], self.w[0]) + np.matmul(input[i-1], self.w[1]) + np.matmul(input[i+1], self.w[2]) + self.b) + np.matmul(np.matmul(target_onehot[i-1], self.w_edge), target_onehot[i])
        
        # i = seq_length - 2
        s = s + np.matmul(target_onehot[seq_length - 2], np.matmul(input[seq_length -2], self.w[0]) + np.matmul(input[seq_length - 1], self.w[2]) + self.b)
        # return alpha/beta tables
        pass

        # compute loss

    def backward(self):
        for i in self.seq_length:
            self.dw[0] = 
            self.db = None
        self.dw_edge = None

    def loss(self):
        pass

