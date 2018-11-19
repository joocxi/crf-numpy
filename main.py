from trainer import Optimizer
from crf import CRF
from load_data import load_data, create_train_val_test_split


# global config
data_path = "data/letter.data"
input_size = 128
classes = 26

# init trainer
dataset = load_data(data_path)
data = create_train_val_test_split(dataset)

model = CRF(input_size, classes)
optimizer = Optimizer(model, data)

# train/val/test
optimizer.train()
optimizer.test()
