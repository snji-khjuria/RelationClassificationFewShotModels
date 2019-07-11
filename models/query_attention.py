import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class QueryAttentiveNetwork(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def generate_query_attentive_prototype(self, support, query):
        B, N, K, D        = support.size()
        _, NQ, _          = query.size()
        support           = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)
        query             = query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1)
        support_attention = self.attention_network(support)
        query_attention   = self.attention_network(query)
        instance_scores   = nn.Softmax(nn.Tanh(support_attention*query_attention).sum(-1), dim=-1)
        instance_scores   = instance_scores.unsqueeze(4).expand(-1, -1, -1, -1, D)
        prototypes        = support*instance_scores.sum(3)
        return prototypes


    def compute_distance(self, prototypes, query):
        return self.__dist__(prototypes, query.unsqueeze(2), 3)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query   = self.sentence_encoder(query)  # (B * N * Q, D)
        support = self.drop(support)
        query   = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query   = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B = support.size(0)  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        #(B, NQ, N, D)
        prototypes = self.generate_query_attentive_prototype(support, query)
        logits = -self.compute_distance(prototypes, query)

        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred