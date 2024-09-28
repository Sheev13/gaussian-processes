import torch
from scipy.io.arff import loadarff

def _class_int_from_string(input_string: str):
    """helper function for 'anana_arff_to_tensor' to obtain the classes as integers"""
    for c in input_string:
        if c.isnumeric():
            return int(c)

def banana_arff_to_tensor(arff_path: str):
    """A function specific to the banana classification dataset that converts it from the
    .arff file format to a torch tensor"""
    l = list(loadarff(arff_path))[0]
    X = torch.tensor([[i[0], i[1]] for i in l])
    class_strings = [str(j[2]) for j in l]
    t = torch.tensor([_class_int_from_string(class_string) - 1 for class_string in class_strings])
    return X, t

def choose_m_from_n(n: int, m: int):
    """function to choose a random subset of m integers from the integers up to n.
    Used in the train-test split functions"""
    assert m <= n
    idx = torch.randperm(n)
    return idx[:m], idx[m:]