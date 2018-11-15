import numpy as np


class Trainer(object):
    def __init__(self, model, data,
                learning_rate=0.0001,
                num_epochs=1000,
                batch_size=32,
                verbose=True):
        self.model = model
        self.data = data

    def train():
        # TODO: 
        steps = None

        for step in range(steps):
            # get data
            batch_data = None
            batch_label = None
            loss = self.model.forward()
            self.model.backward()
            self.model.update()

