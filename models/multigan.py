#creating the metagan model
import torch
import torch.nn as nn

import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F







class MetaGenerator(fewshot_re_kit.multiadversarial_framework.FewShotAdversarialREModel):
    def __init__(self, input_size, N, K, D=230):
        fewshot_re_kit.multiadversarial_framework.FewShotAdversarialREModel.__init__(self)
        self.generator_model = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, N*K*D),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.generator_model(x)
        return x


class MetaDisc(fewshot_re_kit.multiadversarial_framework.FewShotAdversarialREModel):
    def __init__(self, hidden_size=230, relnet_features=230*2):
        fewshot_re_kit.multiadversarial_framework.FewShotAdversarialREModel.__init__(self)
        self.hidden_size = hidden_size
        self.drop=nn.Dropout()
        self.relation_network = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(relnet_features , 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1),
            #TODO: Add the sigmoid layer if you want to
            )

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def euclidean_similarity(self, S, Q):
        distance = self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
        return distance
        #return torch.div(1, 1+distance)
    def relation_score(self, support, query):
        return self.euclidean_similarity(support, query)
        #return self.__batch_dist__(support, query)
        #print("support is ", support.size())
        #print("q query is ", query.size())
        _, nq, _ = query.size()
        B, nc, D = support.size()
        s_s = support.unsqueeze(1).expand(-1, nq, -1, -1)
        q_q = query.unsqueeze(2).expand(-1, -1, nc, -1)
        #cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        #return cos(s_s, q_q)

        nn_input  = torch.cat([s_s, q_q], 3)
        nn_input = nn_input.view(B*nq*nc, -1)
        nn_out = self.relation_network(nn_input)
        nn_out = nn_out.view(B, nq, nc, 1).squeeze(3)
        return nn_out




    def forward(self, support, query, N, K, NQ, is_train=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        "support:(B*N*K, D) and query=(B*NQ, D)"
        eps = 1e-6
        support = self.drop(support)
        query   = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query   = query.view(-1, NQ, self.hidden_size)  # (B, NQ, D)
        B  = support.size(0)  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        # Prototypical Networks
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = -self.relation_score(support, query)
        #if is_train==True:
        #    logits[:, :, N-1]=0
        #print("logits are ", logits)
        #smax = nn.Softmax(2)
        #logits = smax(logits)
        #denominator = torch.sum(torch.exp(logits+eps), 2).unsqueeze(2)
        #denominator = denominator + 1e-3
        #logits  = torch.div(torch.exp(logits+eps), denominator)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred