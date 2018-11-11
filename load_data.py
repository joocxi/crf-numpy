import numpy as np
import pdb

np.random.seed(2018)

letters = 'abcdefghijklmnopqrstuvwxyz'
input_size = 128
lengths = [5502, 688, 687]

def load_data(file_path):
    with open(file_path, "r") as fp:
        data = []
        label = []
        word = []
        char_idx = []
        for line in fp:
            tokens = line.split()
            idx = letters.find(tokens[1])
            word.append(tokens[6:])
            char_idx.append(idx)
            if tokens[2] == "-1":
                data.append(word) 
                label.append(char_idx)
                word = []
                char_idx = []

        all_data = [np.array(word, dtype=np.int32) for word in data]
        all_label = [np.array(char_idx, dtype=np.int32) for char_idx in label]
    dataset = {"data": all_data, "label": all_label}
    return dataset


def create_train_val_test_split(dataset):

    x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
    x = dataset["data"]
    y = dataset["label"]

    permutations = np.random.permutation(len(x))
    train_valid_split = lengths[0]
    valid_test_split = lengths[0] + lengths[1]
    sample_idx = 0
    for i in permutations:
        if sample_idx < train_valid_split:
            x_train.append(x[i])
            y_train.append(y[i])
        elif sample_idx < valid_test_split:
            x_val.append(x[i])
            y_val.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
        sample_idx += 1

    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    dataset = load_data("data/letter.data")
    x_train, _, _, _, _, y_test = create_train_val_test_split(dataset)
