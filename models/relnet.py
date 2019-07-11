#create the relnet network where we have distance metric as neural network for calculation

import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class Relnet(fewshot_re_kit.framework.FewShotRelNetREModel):
    def __init__(self, sentence_encoder, hidden_size=230, relnet_features=460):
        fewshot_re_kit.framework.FewShotRelNetREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(relnet_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relation_network = nn.Sequential(
            nn.Linear(relnet_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            # nn.Sigmoid()
            # nn.Linear(relnet_features, hidden_size),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1)
        )

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def relation_score(self, S, Q):
        _, nq, _ = Q.size()
        B, nc, D = S.size()
        s_s = S.unsqueeze(1).expand(-1, nq, -1, -1)
        q_q = Q.unsqueeze(2).expand(-1, -1, nc, -1)
        nn_input = torch.cat([s_s, q_q], 3)
        nn_input = nn_input.view(B * nq * nc, -1)
        nn_out = self.relation_network(nn_input)  # (B, NQ, C, 1)
        nn_out = nn_out.view(B, nq, nc, 1).squeeze(3)
        return nn_out

    def relationalaaa_score(self, S, Q):
        _, nq, _ = Q.size()
        B, nc, D = S.size()
        # S=(B, C, D) to S=(B, NQ, C, D)
        s_s = S.unsqueeze(1).expand(-1, nq, -1, -1)
        # Q=(B, NQ, D) to Q=(B, NQ, C, D)
        q_q = Q.unsqueeze(2).expand(-1, -1, nc, -1)
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        return cos(s_s, q_q)
        nn_input = torch.cat([s_s, q_q], 3)
        nn_out = self.relation_network(nn_input)  # (B, NQ, C, 1)
        nn_out = torch.sum(nn_out, 3)
        nn_out = nn_out.view(B, nq, nc)
        return nn_out

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = self.drop(support)
        query = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B = support.size(0)  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        # Prototypical Networks
        # Prototype:(Batch, N, D) and Query:(Batch, NQ, D)
        # create relation network from there
        support = torch.mean(support, 2)  # Calculate prototype for each class
        # smax = nn.Softmax(2)
        logits = self.relation_score(support, query)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred