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
        self.transitions.data[:, self.bos_tag_id] = -10000.0
        # no transition allowed from the end of the sentence
        self.transitions.data[self.eos_tag_id, :] = -10000.0

    def forward(self, emissions, tags, mask=None):
        """
        Compute the negative log-likelihood. See log_likelihood method -> -log(p(y|X))
        """
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """
        log(p(y|X))
        Compute the probability of a sequence of tags given a sequence of emissions scores

        Emissions(torch.Tensor):Sequence of emissions for each label.
        IF batch_first(whether the first dimension represents the batch_size) == True -> Shape == (batch_size, seq_len, nb_labels);
        ELSE Shape == (seq_len, batch_size, nb_labels)

        Tags(torch.LongTensor): Sequence of labels
        IF batch_first == True -> Shape == (batch_size, seq_len)
        ELSE Shape == (seq_len, batch_size)

        Mask (torch.FloatTensor, optional): Tensor representing valid positions.
        Ignore computation associated with pad symbol(Not valid position -> it is PAD position)
        Example of mask matrix: ["Hi", "I", "am", "Pavel", <PAD>, <PAD>] -> [1,1,1,1,0,0]
        IF None -> all positions are considered valid.
        IF batch_first == True -> Shape == (batch_size, seq_len)
        ELSE Shape == (seq_len, batch_size)

        Return:
            torch.Tensor: the log-likelihood for each sequence in the batch
            Shape == (batch_size,)


        """
        if not self.batch_first:
            # swap dimension 0 and 1
            # transpose Shape == (seq_len, batch_size, nb_labels)
            # to (batch_size, seq_len, nb_labels)
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            # all positions are considered valid
            # initialize tensor with 1 in each position with the given shape
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        # d(L)/d(Lambda) = sum(f(x,y)) - sum(f(x,y)*p(y|x))
        return torch.sum(scores - partition)

    def compute_scores(self, emissions, tags, mask):
        """"
        Compute the scores for a given batch of emissions with their tags
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags(torch.LongTensor): (batch_size, seq_len)
            mask(torch.FloatTensor):(batch_size, seq_len)
        Returns:
            torch.Tensor:Scores for each batch
            Shape of (batch_size, )
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)

        # save first and last tags
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()


crf = CRF(3, 0, 2)
print(crf.nb_labels)
print(crf.transitions)
print(torch.ones(10))
print(crf.transitions.transpose(0, 1))
