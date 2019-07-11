def train(self,
          model,
          model_name,
          B, N_for_train, N_for_eval, K, Q,
          ckpt_dir='./checkpoint',
          test_result_dir='./test_result',
          learning_rate=1e-6,
          lr_step_size=20000,
          weight_decay=1e-4,
          train_iter=30000,
          val_iter=200,
          val_step=500,
          test_iter=3000,
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
        
    print("Start training...")

    # Init
    parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
    if pretrain_model:
        checkpoint = self.__load_model__(pretrain_model)
        model.load_state_dict(checkpoint['state_dict'])
        start_iter = checkpoint['iter'] + 1
    else:
        start_iter = 0

    if cuda:
        model = model.cuda()
    model.train()

    # Training
    best_acc = 0
    not_best_count = 0  # Stop training after several epochs without improvement.
    iter_loss = 0.0
    iter_right = 0.0
    iter_sample = 0.0
    for it in range(start_iter, start_iter + train_iter):
        scheduler.step()
        support, query, label = self.train_data_loader.next_batch(B, N_for_train, K, Q)
        logits, pred = model(support, query, N_for_train, K, Q)
        onehot_label = self.make_one_hot(label, N_for_train)
        loss = model.loss(logits, onehot_label)
        right = model.accuracy(pred, label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(parameters_to_optimize, 10)
        optimizer.step()

        iter_loss += self.item(loss.data)
        iter_right += self.item(right.data)
        iter_sample += 1
        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                                    100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if it % val_step == 0:
            iter_loss = 0.
            iter_right = 0.
            iter_sample = 0.

        if (it + 1) % val_step == 0:
            acc = self.do_eval(model, B, N_for_eval, K, Q, val_iter)
            test_acc = self.do_eval(model, B, N_for_eval, K, Q, val_iter, False)
            print("Validation accuracy is ", acc)
            print("Testing accuracy is ", test_acc)
            model.train()
            if acc > best_acc:
                print('Best checkpoint')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                torch.save({'state_dict': model.state_dict()}, save_path)
                best_acc = acc

    print("\n####################\n")
    print("Finish training " + model_name)
    test_acc = self.eval(model, B, N_for_eval, K, Q, test_iter,
                         ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
    print("Test accuracy: {}".format(test_acc))


import  torch
from torch.autograd import Variable

def make_one_hot(labels, C):
    labels = labels.view(-1, 1)
    # y_onehot = torch.cuda.FloatTensor(labels.size(0), C)
    y_onehot = torch.FloatTensor(labels.size(0), C)
    y_onehot.zero_()
    print(y_onehot)
    y_onehot.scatter_(1, labels, 1)
    return y_onehot



#create one hot labels that are present in there for the answer
#change the model to return the sigmoid function
#convert the loss function to return the mean square loss
#try to produce the training of the model for the final answers