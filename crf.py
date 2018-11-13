import numpy as np


class CRF(object):

    def __init__(self,
                input_size,
                classes):
        self.input_size = input_size
        self.classes = classes
 
    def ready(self):
        self.alpha = np.zeros((0, 0))
        self.beta = np.zeros((0, 0))
        self.w = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]
        self.b = np.zeros((self.classes))

        self.w_edge = np.zeros((self.classes, self.classes))

        self.dw = [np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes)),
                    np.zeros((self.input_size, self.classes))]

        self.db = [np.zeros((self.classes))]
        self.dw_edge = np.zeros((self.classes, self.classes))


    def forward(self, input, target):
        """
        input, (seq_length, input_size)
        target, (seq_length)
        """
        self.seq_length = input.shape[0]
        self.target_onehot = np.zeros((self.seq_length, self.classes))
        self.target_onehot[np.arange(self.seq_length), target] = 1

        self.alpha = np.zeros((self.classes, self.seq_length))
        self.beta = np.zeros((self.classes, self.seq_length + 1))

        s = 0
        s = s + np.matmul(self.target_onehot[0], np.matmul(input[0], self.w[0]) + np.matmul(input[1], self.w[2]) + self.b)

        for i in range(1, self.seq_length - 1):
            s = s + np.matmul(self.target_onehot[i], np.matmul(input[i], self.w[0]) + np.matmul(input[i-1], self.w[1]) + np.matmul(input[i+1], self.w[2]) + self.b) + np.matmul(np.matmul(self.target_onehot[i-1], self.w_edge), self.target_onehot[i])
        
        # i = seq_length - 1
        s = s + np.matmul(self.target_onehot[self.seq_length - 1], np.matmul(input[self.seq_length - 1], self.w[0]) + np.matmul(input[self.seq_length - 2], self.w[1]) + self.b) + np.matmul(np.matmul(self.target_onehot[self.seq_length - 2], self.w_edge), self.target_onehot[self.seq_length - 1])

        # shape: (self.classes, ) = (input_size, )*(input_size, classes)
        self.alpha[:, 0] = np.exp(np.matmul(input[0], self.w[0]) + np.matmul(input[1], self.w[2]))

        # shape: (self.classes)
        for i in range(1, self.seq_length-1):
            #clique_potential = np.matmul
            xw = (np.matmul(input[i], self.w[0]) + np.matmul(input[i+1], self.w[2]) + np.matmul(input[i-1], self.w[1]))
            unary = xw[:, None]*self.target_onehot[i][None, :]
            clique_potential = unary + self.w_edge
            self.alpha[:, i] = np.matmul(np.exp(clique_potential), self.alpha[:, i-1])

        xw = unary = (np.matmul(input[i], self.w[0]) + np.matmul(input[i+1], self.w[2]) + np.matmul(input[i-1], self.w[1]))
        unary = xw[:, None]*self.target_onehot[self.seq_length-1]
        clique_potential = unary + self.w_edge
        self.alpha[:, self.seq_length-1] = np.matmul(np.exp(clique_potential), self.alpha[:, self.seq_length-2]) 

        #for i in range(1, self.seq_length):
        #    for j in range(self.classes):
        #        self.alpha[j, i] = self.alpha[0, i-1]*np.exp(np.matmul(

        self.beta[:, self.seq_length - 1] = 1
        xw = (np.matmul(input[self.seq_length-1], self.w[0]) + np.matmul(input[self.seq_length-2], self.w[1]))
        unary = xw[:, None]*self.target_onehot[self.seq_length - 1][None, :]
        clique_potenial = unary + xw
        self.beta[:, self.seq_length -2] = np.matmul(self.beta[:, self.seq_length-1], np.exp(clique_potential.T))
        for i in range(self.seq_length - 3, -1, -1):
            xw = (np.matmul(input[i+1], self.w[0]) + np.matmul(input[i+2], self.w[2]) + np.matmul(input[i], self.w[1]))
            unary = xw[:, None]*self.target_onehot[i+1][None, :]

            clique_potenial = unary + xw
            self.beta[:, i] = np.matmul(self.beta[:, i+1], np.exp(clique_potential.T))

        # compute loss
        Z = np.sum(self.alpha[:, -1])
        loss = Z - s
        return loss


    def backward(self, input, target):
        # dw: (input_size, classes)
        # input: (seq_length, input_size)
        # self.target_onehot: (seq_length, classes)
        
        # shape: (input_size, classes)
        np.matmul(input.T, self.target_onehot)

        # shape: (input_size, classes)
        for i in range(self.seq_length):
            for j in range(self.classes):
                self.dw[i, j]=None
        self.dw_edge = None

    def loss(self):
        pass

