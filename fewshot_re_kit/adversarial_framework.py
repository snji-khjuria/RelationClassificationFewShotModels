import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from itertools import chain
from numpy.random import uniform


class FewShotAdversarialREModel(nn.Module):
    def __init__(self):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        #self.cost = nn.NLLLoss()
        self.cost = nn.CrossEntropyLoss()
        #self.cost = nn.NLLLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def disc_loss(self, logits, label):
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def gen_loss(self, logits, label):
        N = logits.size(-1)
        return -1*self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotAdversarialREFramework:
    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def create_task_stats(self, support, B, N, K, D):
        support = support.view(B, 1, N * K, D)
        max_pool_op = nn.AvgPool2d((K, 5), (K, 2))
        support = max_pool_op(support)
        task_stats = support.view(B, -1)
        noise_vector = self.noise(B)
        gen_input = torch.cat([task_stats, noise_vector], 1)
        return gen_input

    def noise(self, size):
        n = torch.FloatTensor(size, 200).normal_().cuda()
        # n = Variable(torch.cuda.randn(size, 200))
        return n

    def produce_query_set(self, generator, support, N, K, D):
        support = support.view(-1, N, K, D)
        B, _, _, _ = support.size()
        gen_input = self.create_task_stats(support, B, N, K, D)
        gen_output = generator(gen_input)
        gen_output = gen_output.view(B, K, D)
        return gen_output

    def augment_support_set(self, generator, support, N, K, D):
        support = support.view(-1, N, K, D)
        #print("Support is ", support.size())
        support_saved = support
        B, _, _, _ = support.size()
        gen_input = self.create_task_stats(support, B, N, K, D)
        gen_output = generator(gen_input)
        gen_output = gen_output.view(B, 1, K, D)
        resulting_support = torch.cat([support_saved, gen_output], 1)
        resulting_support = resulting_support.view(-1, D)
        return resulting_support

    def train_disc(self, sentence_encoder, support, query, gen_model, disc_model, N_for_train, K, Q):
        encoded_support = sentence_encoder(support)
        _, D = encoded_support.size()
        #print("encoded support set size is ", encoded_support.size())
        #import sys
        #sys.exit(1)
        augmented_support = self.augment_support_set(gen_model, encoded_support, N_for_train, K, D)
        encoded_query = sentence_encoder(query)
        # print("encoded support set is ", encoded_support.size())
        # print("augmented support set is ", augmented_support.size())
        # print("encoded query is ", encoded_query.size())
        # print("Batch size is ", B)
        # print("size K is ", K)
        # print("Dim is ", D)
        # print("N before is ", N_for_train)
        N_train = N_for_train + 1
        N_queries = N_for_train * Q
        # print("training with classes ", N_train)
        # print("training with queries ", N_queries)
        logits, pred = disc_model(augmented_support, encoded_query, N_train, K, N_queries, True)
        return logits, pred
        # print("logits size is ", logits.size())

    def train_gen(self, sentence_encoder, support, query, gen_model, disc_model, N_for_train, K):
        gen_support_encoded = sentence_encoder(support)
        _, D = gen_support_encoded.size()
        augmented_support = self.augment_support_set(gen_model, gen_support_encoded, N_for_train, K, D)
        fake_queries = self.produce_query_set(gen_model, gen_support_encoded, N_for_train, K, D)
        gen_logits, gen_pred = disc_model(augmented_support, fake_queries, N_for_train + 1, K, K, True)
        return gen_logits, gen_pred

    def train2(self, disc_model, gen_model, sentence_encoder,
               model_name,
               B, N_for_train, N_for_eval, K, Q, disc_learning_rate,
               ckpt_dir='./checkpoint',
               test_result_dir='./test_result',
               learning_rate=1e-1,
               lr_step_size=20000,
               weight_decay=1e-5,
               train_iter=30000,
               val_iter=100,
               val_step=200,
               test_iter=3000,
               gen_learning_rate=1e-6,
               cuda=True,
               pretrain_model=None,
               optimizer=optim.SGD):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        for i in range(100):
            # learning_rate = 10** uniform(-6, -3)
            # Init
            disc_parameters = chain(disc_model.parameters(), sentence_encoder.parameters())
            disc_parameters_to_optimize = filter(lambda x: x.requires_grad, disc_parameters)
            gen_parameters_to_optimize = filter(lambda x: x.requires_grad, gen_model.parameters())
            disc_optimizer = optim.Adam(disc_parameters_to_optimize, disc_learning_rate)
            gen_optimizer = optim.Adam(gen_parameters_to_optimize, gen_learning_rate)
            start_iter = 0
            if cuda:
                disc_model = disc_model.cuda()
                gen_model = gen_model.cuda()
            disc_model.train()
            gen_model.train()
            # Training
            best_acc = 0
            not_best_count = 0  # Stop training after several epochs without improvement.
            # iter_loss = 0.0
            # iter_right = 0.0
            iter_sample = 0.0
            disc_iter_loss = 0.0
            disc_iter_right = 0.0
            gen_iter_loss = 0.0
            gen_iter_right = 0.0
            for it in range(start_iter, start_iter + train_iter):
                support, query, label_disc = self.train_data_loader.next_batch(B, N_for_train, K, Q)
                logits_disc, pred_disc = self.train_disc(sentence_encoder, support, query, gen_model, disc_model,
                                                         N_for_train, K, Q)
                disc_loss = disc_model.disc_loss(logits_disc, label_disc)
                disc_right = disc_model.accuracy(pred_disc, label_disc)
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                disc_iter_loss += self.item(disc_loss.data)
                disc_iter_right += self.item(disc_right.data)
                iter_sample += 1

                gen_support, gen_query, label_gen = self.train_data_loader.next_batch(B, N_for_train, K, Q)
                # label_gen = produce_labels_for_generator(B, K)
                label_gen = torch.ones((B, K)).new_full((B, K), N_for_train, dtype=torch.int64).cuda()
                logits_gen, pred_gen = self.train_gen(sentence_encoder, support, query, gen_model, disc_model,
                                                      N_for_train,
                                                      K)
                gen_loss = gen_model.gen_loss(logits_gen, label_gen)
                gen_right = gen_model.accuracy(pred_gen, label_gen)
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                gen_iter_loss += self.item(gen_loss.data)
                gen_iter_right += self.item(gen_right.data)
                if it >= 400:
                    break

            print('learning_rate: {} | disc loss: {:f}, gen loss: {:f} accuracy: {:f}%'.format(disc_learning_rate,
                                                                                               disc_iter_loss / iter_sample,
                                                                                               gen_iter_loss / iter_sample,
                                                                                               100 * disc_iter_right / iter_sample) + '\r')
            return
    #0.00044487909314557
    def train(self, disc_model, gen_model, sentence_encoder,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              ckpt_dir='./checkpoint',
              test_result_dir='./test_result',
              learning_rate=1e-1,
              lr_step_size=30000,
              weight_decay=1e-5,
              train_iter=20000,
              val_iter=1000,
              val_step=300,
              test_iter=3000,
              disc_learning_rate=1e-4,
              gen_learning_rate=1e-12,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.SGD):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")

        # Init
        disc_parameters = chain(disc_model.parameters(), sentence_encoder.parameters())
        disc_parameters_to_optimize = filter(lambda x: x.requires_grad, disc_parameters)
        gen_parameters_to_optimize = filter(lambda x: x.requires_grad, gen_model.parameters())
        disc_optimizer = optim.Adam(disc_parameters_to_optimize, disc_learning_rate, amsgrad=True,
                                    weight_decay=weight_decay)
        gen_optimizer = optim.Adam(gen_parameters_to_optimize, gen_learning_rate, amsgrad=True,
                                   weight_decay=weight_decay)
        start_iter = 0
        if cuda:
            disc_model = disc_model.cuda()
            gen_model = gen_model.cuda()
        disc_model.train()
        gen_model.train()
        # Training
        best_acc = 0
        not_best_count = 0  # Stop training after several epochs without improvement.
        # iter_loss = 0.0
        # iter_right = 0.0
        iter_sample = 0.0
        disc_iter_loss = 0.0
        disc_iter_right = 0.0
        gen_iter_loss = 0.0
        gen_iter_right = 0.0
        with autograd.detect_anomaly():
            for it in range(start_iter, start_iter + train_iter):
                support, query, label_disc = self.train_data_loader.next_batch(B, N_for_train, K, Q)
                logits_disc, pred_disc = self.train_disc(sentence_encoder, support, query, gen_model, disc_model,
                                                         N_for_train, K, Q)
                disc_loss = disc_model.disc_loss(logits_disc, label_disc)
                disc_right = disc_model.accuracy(pred_disc, label_disc)
                disc_optimizer.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm(disc_parameters_to_optimize, 10)
                disc_optimizer.step()

                disc_iter_loss += self.item(disc_loss.data)
                disc_iter_right += self.item(disc_right.data)
                iter_sample += 1

                gen_support, gen_query, label_gen = self.train_data_loader.next_batch(B, N_for_train, K, Q)
                # label_gen = produce_labels_for_generator(B, K)
                label_gen = torch.ones((B, K)).new_full((B, K), N_for_train, dtype=torch.int64).cuda()
                logits_gen, pred_gen = self.train_gen(sentence_encoder, support, query, gen_model, disc_model,
                                                      N_for_train, K)
                gen_loss = gen_model.gen_loss(logits_gen, label_gen)
                gen_right = gen_model.accuracy(pred_gen, label_gen)
                gen_optimizer.zero_grad()
                gen_loss.backward()
                nn.utils.clip_grad_norm(gen_parameters_to_optimize, 10)
                gen_optimizer.step()

                gen_iter_loss += self.item(gen_loss.data)
                gen_iter_right += self.item(gen_right.data)
                sys.stdout.write('step: {} | disc loss: {:f}, gen loss: {:f} accuracy: {:f}%'.format(it + 1,
                                                                                                     disc_iter_loss / iter_sample,
                                                                                                     gen_iter_loss / iter_sample,
                                                                                                     100 * disc_iter_right / iter_sample) + '\r')
                # sys.stdout.write('step: {0:4} | disc loss: {1:2.6f}, gen loss: {1:2.6f} accuracy: {2:3.2f}%'.format(it + 1, disc_iter_loss / iter_sample, gen_iter_loss / iter_sample,
                #                                                                            100 * disc_iter_right / iter_sample) + '\r')
                sys.stdout.flush()
                if it % val_step == 0:
                    disc_iter_loss = 0.
                    disc_iter_right = 0.
                    gen_iter_loss = 0.
                    gen_iter_right = 0.
                    iter_sample = 0.

                if (it + 1) % val_step == 0:
                    acc = self.do_eval(disc_model, sentence_encoder, B, N_for_eval, K, Q, val_iter)
                    testing_acc = self.do_eval(disc_model, sentence_encoder, B, N_for_eval, K, Q, val_iter, True)
                    print("Validation acc: ", acc)
                    print("Testing acc: ", testing_acc)
                    disc_model.train()
                    if acc > best_acc:
                        print('Best checkpoint')
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                        torch.save({'state_dict': disc_model.state_dict()}, save_path)
                        best_acc = acc

        print("\n####################\n")
        print("Finish training " + model_name)
        test_acc = self.eval(disc_model, B, N_for_eval, K, Q, test_iter,
                             ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
        print("Test accuracy: {}".format(test_acc))

    def do_eval(self,
             model, sentence_encoder,
             B, N, K, Q,
             eval_iter, testing=False,
             ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        model.eval()
        if testing==False:
            eval_dataset = self.val_data_loader
        else:
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, label = eval_dataset.next_batch(B, N, K, Q)
            support = sentence_encoder(support)
            query = sentence_encoder(query)
            # print("support set is ", support.size())
            # print("query set is ", query.size())
            logits, pred = model(support, query, N, K, N * Q)
            right = model.accuracy(pred, label)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write(
                '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        print("")
        return iter_right / iter_sample
    def eval(self,
             model, sentence_encoder,
             B, N, K, Q,
             eval_iter,
             ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, label = eval_dataset.next_batch(B, N, K, Q)
            support = sentence_encoder(support)
            query = sentence_encoder(query)
            # print("support set is ", support.size())
            # print("query set is ", query.size())
            logits, pred = model(support, query, N, K, N * Q)
            right = model.accuracy(pred, label)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write(
                '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        print("")
        return iter_right / iter_sample