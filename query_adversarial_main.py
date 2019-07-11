import models
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
import sys
from fewshot_re_kit.adversarial_framework import FewShotAdversarialREFramework
from models.metagan_queryattentive import MetaDisc, MetaGenerator
import os
import torch
torch.cuda.set_device(1)
model_name = 'metagan'
N = 5
K = 5
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader   = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader  = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)

framework        = FewShotAdversarialREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
sentence_encoder.cuda()
print("Makign the adversarial model for training....")
disc_model = MetaDisc()
gen_model = MetaGenerator(6980, K)
model = disc_model
gen_model.cuda()
disc_model.cuda()
print("Model is ready.")
framework.train(disc_model, gen_model, sentence_encoder, model_name, 4, 60, 16, K, 5)
print("Modeling is done...")