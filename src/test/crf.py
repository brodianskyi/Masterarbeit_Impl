import torch
from torch import nn


class CRF(nn.Module):

    """
    Implementation of Linear-chain Conditional Random Field (CRF).
    """

    def __init__(self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True):
        super().__init__()
        # number of labels in tagset, including special symbols
        self.nb_labels = nb_labels
        # integer representing the beginning of sentence symbol in tagset
        self.bos_tag_id = bos_tag_id
        # integer representing the end of sentence
        self.eos_tag_id = eos_tag_id
        # whether the first dimension represents the batch dimension
        self.batch_first = batch_first
        # Parameter containing: tensor([ [0.1, 0.2], [0.3, 0.4] ] -> [kol-vo of labels]
        # requires_grad=True -> if the gradient is required
        # Create a tensor (n_label, n_label)
        # transitions between labels
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # initialize with the big negative number
        # so exp(-10000) will tend to zero
        # no transitions allowed to the beginning of sentence
        # [индекс с какой строки начинать: какой заканчивать , с какого столбца в этой строке начинать:каким закончить]
        # во всех строках каждый столбец с номером bos_tag_id = -10000.0
        self.transitions.data[:, self.bos_tag_id] = -10000.0
        # no transition allowed from the end of the sentence
        # в строке с номером eos_tag_id все стобцы = -10000.0
        self.transitions.data[self.eos_tag_id, :] = -10000.0
        '''
        where bos_tag_id = 0; 
              eos_tag_id = 2.
        tensor([[-1.0000e+04, -4.4406e-02,  7.9665e-02],
                [-1.0000e+04,  3.2845e-02, -1.1155e-03],
                [-1.0000e+04, -1.0000e+04, -1.0000e+04]], requires_grad=True)
        '''


crf = CRF(3, 0, 2)
print(crf.nb_labels)
print(crf.transitions)
