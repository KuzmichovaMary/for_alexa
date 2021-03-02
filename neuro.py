import numpy as np
from scipy.special import expit


class Neuronet:

    def __init__(self, in_nodes, *hid_nodes, out_nodes, l_rate):
        self.in_nodes = in_nodes
        self.hid_nodes = hid_nodes
        self.out_nodes = out_nodes
        self.l_rate = l_rate
        self.w0 = ...
        self.w1 = ...

    def activation(self, x):
        return expit(x)

    def prediction(self, inputs_list):
        inputs = ...
        hid_outputs = ...
        out_outputs = ...
        return out_outputs