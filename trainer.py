import numpy as np
from tqdm import tqdm


class Optimizer(object):
    def __init__(self, model, data,
                learning_rate=0.0001,
                num_epochs=200,
                batch_size=32,
                val_period=20,
                verbose=True):
        self.model = model
        self.model.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_period = val_period
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data

    def train(self):

        for epoch in range(self.num_epochs):
            print("epoch %i starting..." % epoch)
            losses = []
            for input, target in tqdm(zip(self.x_train, self.y_train), total=len(self.y_train)):
                loss = self.model.forward(input, target)
                losses.append(loss)
                self.model.backward(input, target)
                self.model.update()

            print ("loss of epoch %i is: %f " % (epoch, np.mean(losses)))
            
            # get error on val set
            if epoch % self.val_period == 0:
                self.validate()


    def validate(self):
        # TODO: 
        errors = []
        for input, target in tqdm(zip(self.x_val, self.y_val), total=len(self.y_val)):
            pred = self.model.predict(input)
            class_error = np.equal(pred, target).astype(np.int32)
            errors.append(class_error)

        errors = np.hstack(errors)
        errors = np.mean(errors)
        print ("validation error: %f" % errors)


    def test(self):
        errors = []
        for input, target in tqdm(zip(self.x_test, self.y_test), total=len(self.y_test)):
            pred = self.model.predict(input)
            class_error = np.equal(pred, target).astype(np.int32)
            errors.append(class_error)

        errors = np.hstack(errors)
        errors = np.mean(errors)
        print ("test error: %f" % errors)
