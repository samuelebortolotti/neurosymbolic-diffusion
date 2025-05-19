from typing import Optional
import torch
import torch.nn as nn


"""
class BOIAConceptizer:
    network definitions of encoder and decoder using fully connected network
    encoder c() is the network by computing concept c(e(x))
def __init__:
    define parameters (e.g., # of layers) 
def encode:
    compute concepts
"""


class BOIAConceptizer(nn.Module):
    """
    def __init__:
        define parameters (e.g., # of layers)
        MLP-based conceptizer for concept basis learning.
    Inputs:
        din (int): input size
        nconcept (int): # of all concepts
    Return:
        None
    """

    def __init__(self, din, nconcept):
        super(BOIAConceptizer, self).__init__()
        

        # set self hyperparameters
        self.din = din  # Input dimension
        self.nconcept = nconcept  # Number of "atoms"/concepts

        """
        encoding
        self.enc1: encoder for known concepts
        """
        self.enc1 = nn.Linear(self.din, self.nconcept)

    """ 
    def forward:
        compute concepts
    Inputs:
        x: output of pretrained model (encoder)
    Return:
        encoded_1: predicted known concepts
    """

    def forward(self, x, one_hot_w_SBWD: Optional[torch.Tensor] = None):
        if one_hot_w_SBWD is not None:
            one_hot_w_SBWtD = one_hot_w_SBWD.view(one_hot_w_SBWD.shape[:-2] + (-1,))
            p = torch.cat((x, one_hot_w_SBWtD), dim=-1)
        else:
            # resize
            p = x.view(x.size(0), -1)

        T = 2.5
        logits_c = self.enc1(p) / T

        if one_hot_w_SBWD is not None:
            encoded_1 = logits_c.unsqueeze(-1)
            encoded_false_positive = torch.cat([torch.zeros_like(encoded_1), encoded_1], dim=-1)
            return encoded_false_positive
        return encoded_1

class BOIAConceptizerMLP(nn.Module):
    """
    This model mimics the conceptizer from BEARS https://github.com/samuelebortolotti/bears/blob/master/BDD_OIA/conceptizers_BDD.py
    def __init__:
        define parameters (e.g., # of layers)
        MLP-based conceptizer for concept basis learning.
    Inputs:
        din (int): input size
        nconcept (int): # of all concepts
    Return:
        None
    """

    def __init__(self, din, nconcept, hidden_dim=512):
        super(BOIAConceptizerMLP, self).__init__()


        # set self hyperparameters
        self.din = din  # Input dimension
        self.nconcept = nconcept  # Number of "atoms"/concepts

        """
        encoding
        self.enc1: encoder for known concepts
        """
        self.enc1 = nn.Linear(self.din, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(hidden_dim, self.nconcept)

    """ 
    def forward:
        compute concepts
    Inputs:
        x: output of pretrained model (encoder)
    Return:
        encoded_1: predicted known concepts
    """

    def forward(self, x, one_hot_w_SBWD: Optional[torch.Tensor] = None):
        if one_hot_w_SBWD is not None:
            one_hot_w_SBWtD = one_hot_w_SBWD.view(one_hot_w_SBWD.shape[:-2] + (-1,))
            p = torch.cat((x, one_hot_w_SBWtD), dim=-1)
        else:
            # resize
            p = x.view(x.size(0), -1)

        p = self.dropout(self.relu(self.enc1(p)))
        logits_c = self.classifier(p)

        if one_hot_w_SBWD is not None:
            logits_c = logits_c.unsqueeze(-1)
            # For compatibility with later softmax
            # Since softmax([0, l])[1] = sigmoid(l)
            encoded_softmax_logits = torch.cat([torch.zeros_like(logits_c), logits_c], dim=-1)
            return encoded_softmax_logits
        return logits_c
