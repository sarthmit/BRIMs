# coding: utf-8
import numpy as np
import torch
import random

# Set the random seed manually for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
np.random.seed(0)

import argparse
import time
import math
import os
import torch.nn as nn
import torch.onnx
import datetime
import shutil
import pickle
import rnn_models
import baseline_lstm_model
import random
import mixed
from cifar_seq_data_classify import cifar_data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# same hyperparameter scheme as word-language-model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--algo', type=str, choices=('blocks', 'lstm','mixed'))
parser.add_argument('--num_blocks', nargs='+', type=int, default=[6])
parser.add_argument('--nhid', nargs='+', type=int, default=[300])
parser.add_argument('--topk', nargs='+', type=int, default=[4])
parser.add_argument('--block_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--layer_dilation', nargs='+', type=int, default=-1)

parser.add_argument('--use_inactive', action='store_true',
                    help='Use inactive blocks for higher level representations too')
parser.add_argument('--blocked_grad', action='store_true',
                    help='Block Gradients through inactive blocks')
parser.add_argument('--scheduler', action='store_true',
                    help='Scheduler for Learning Rate')

# parameters for adaptive softmax
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')

# experiment name for this run
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')

args = parser.parse_args()

best_test = {16: 0.0, 19:0.0, 24:0.0, 32:0.0}
best_val = 0.0
best_val_epoch = 0.0

best_test_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}

inp_size = 16
val_size = inp_size

######## Plot Specific Details ########

colors = ['white', 'black']
cmap = LinearSegmentedColormap.from_list('name', colors)
norm = plt.Normalize(0, 1)

matplotlib.rc('xtick', labelsize=7.5)
matplotlib.rc('ytick', labelsize=7.5)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# Get Data Loaders

train_loader, val_loader, test_loader, noise_loader = cifar_data()

# create folder for current experiments
# name: args.name + current time
# includes: entire scripts for faithful reproduction, train & test logs

folder_name = str(datetime.datetime.now())[:-7]
if args.name is not None:
    folder_name = str(args.name)

if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if not os.path.exists(folder_name+'/visuals/'):
    os.mkdir(folder_name+'/visuals/')

logger_args = open(os.path.join(os.getcwd(), folder_name, 'args.txt'), 'a')
logger_output = open(os.path.join(os.getcwd(), folder_name, 'output.txt'), 'a')
logger_epoch_output = open(os.path.join(os.getcwd(), folder_name, 'epoch_output.txt'), 'a')

# save args to logger
logger_args.write(str(args) + '\n')

# define saved model file location
savepath = os.path.join(os.getcwd(), folder_name)

###############################################################################
# Build the model
###############################################################################

ntokens = 256
n_out = 10
if args.adaptivesoftmax:
    print("Adaptive Softmax is on: the performance depends on cutoff values. check if the cutoff is properly set")
    print("Cutoffs: " + str(args.cutoffs))
    if args.cutoffs[-1] > ntokens:
        raise ValueError("the last element of cutoff list must be lower than vocab size of the dataset")
    criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(args.nhid, ntokens, cutoffs=args.cutoffs).to(device)
else:
    criterion = nn.CrossEntropyLoss()


if args.algo == "blocks":
    rnn_mod = rnn_models.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, n_out, args.dropout, False,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive = args.use_inactive,
                            blocked_grad=args.blocked_grad, block_dilation=args.block_dilation,
                            layer_dilation=args.layer_dilation, num_modules_read_input=2).to(device)
elif args.algo == "lstm":
    rnn_mod = baseline_lstm_model.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, n_out, args.dropout, False,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax, cutoffs=args.cutoffs).to(device)
elif args.algo == 'mixed':
    rnn_mod = mixed.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, False,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive=args.use_inactive ,
                            blocked_grad=args.blocked_grad).to(device)
else:
    raise Exception("Algorithm option not found")

if os.path.exists(folder_name+'/model.pt'):
    print("Reloading+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    state = torch.load(folder_name+'/model.pt')
    model.load_state_dict(state['state_dict'])
    global_epoch = state['epoch']
    best_val = state['best_val']
else:
    global_epoch = 1

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Built with Total Number of Trainable Parameters: " + str(total_params))
if not args.cudnn:
    print(
        "--cudnn is set to False. the model will use RNNCell with for loop, instead of cudnn-optimzed RNN API. Expect a minor slowdown.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

###############################################################################
# Load the model checkpoint if specified and restore the global & best epoch
###############################################################################

if args.resume is not None:
    print("--resume detected. loading checkpoint...")
global_epoch = args.resume if args.resume is not None else 0
best_epoch = args.resume if args.resume is not None else 0
if args.resume is not None:
    loadpath = os.path.join(os.getcwd(), "model_{}.pt".format(args.resume))
    if not os.path.isfile(loadpath):
        raise FileNotFoundError(
            "model_{}.pt not found. place the model checkpoint file to the current working directory.".format(
                args.resume))
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_epoch = checkpoint["global_epoch"]
    best_epoch = checkpoint["best_epoch"]

print("Model Built with Total Number of Trainable Parameters: " + str(total_params))


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if args.algo == "lstm":
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)
    hidden = []
    if args.nlayers==1:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)
    for i in range(args.nlayers):
        if isinstance(h[i], torch.Tensor):
            hidden.append(h[i].detach())
        else:
            hidden.append(tuple((h[i][0].detach(), h[i][1].detach())))
    return hidden

def mnist_prep(x, test_upsample=inp_size):
    d = x
    d = F.interpolate(d, size=(test_upsample,test_upsample), mode='nearest')
    d = d.transpose(3,1).transpose(1,2)
    d = d.reshape((d.shape[0],test_upsample*test_upsample,3)) * 255.
    d = d.round().to(dtype=torch.int64)
    d = d.permute(1,0,2)
    return d

def evaluate_(test_lens, split):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    if split is "Val":
        loader = val_loader
    else:
        loader = test_loader

    test_acc = {i: 0.0 for i in test_lens}
    val_loss = 0.0

    for test_len in test_lens:
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for n, (d,t) in zip(noise_loader,loader):
                hidden = model.init_hidden(args.batch_size)

                if split is "Test" and args.noise:
                    d = torch.clamp(d + n/32., min=0.0, max=1.0)

                d = mnist_prep(d, test_upsample = test_len)
                t = t.to(dtype=torch.int64)

                data = Variable(d.cuda())
                targets = Variable(t.cuda())

                num_batches += 1

                output, hidden = model(data, hidden)

                if not args.adaptivesoftmax:
                    loss = criterion(output[-1], targets)
                    acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
                else:
                    _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)

                total_acc += acc.item()
                hidden = repackage_hidden(hidden)
                if test_len is inp_size:
                    val_loss += loss.item()

        test_acc[test_len] = total_acc / num_batches

    if split is "Val":
        val_loss = val_loss / num_batches
        if args.scheduler:
            scheduler.step(val_loss)

    return test_acc

def train(epoch):
    global best_val, best_val_epoch
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()

    i = 0
    j = 0

    calc_mask = True

    test_epoch = {16: 0.0,19:0.0, 24:0.0, 32:0.0}
    val_epoch = 0.0

    for d,t in train_loader:
        hidden = model.init_hidden(args.batch_size)
        model.train()
        i += 1

        d = mnist_prep(d, test_upsample=inp_size)
        t = t.to(dtype=torch.int64)

        data = Variable(d.cuda())
        targets = Variable(t.cuda())

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        forward_start_time = time.time()

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden, calc_mask)

        if not args.adaptivesoftmax:
            loss = criterion(output[-1], targets)
            acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
        else:
            raise Exception('not implemented')
            _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
        total_loss += acc.item()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | average acc {:5.4f} | ppl {:8.2f}'.format(
                epoch, i, len(train_loader.dataset) // args.batch_size, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, forward_elapsed_time * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss))
            # print and save the log
            print(printlog)
            logger_output.write(printlog + '\n')
            logger_output.flush()
            total_loss = 0.
            # reset timer
            start_time = time.time()
            forward_start_time = time.time()
            forward_elapsed_time = 0.

        if i % args.log_interval == 0 and i > 0 and epoch%10==0:
            j += 1
            test_lens = [16, 19,24,32]

            test_acc = evaluate_(test_lens, split="Test")
            val_acc = evaluate_([val_size], split="Val")[val_size]

            printlog = ''

            if val_acc > best_val:
                for key in test_acc:
                        best_val = val_acc
                        best_test[key] = test_acc[key]
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val' : best_val
                    }
                torch.save(state, folder_name+'/best_model.pt')

            for key in test_acc:
                test_epoch[key] += test_acc[key]
                printlog = printlog + '\n' + '|Seq_len: {} | Test Current: {} | Test Optim: {} | Val Current: {} | Val Best: {} |'.format(str(key), str(test_acc[key]), str(best_test[key]), str(val_acc), str(best_val))

            val_epoch += val_acc

            logger_output.write(printlog+'\n\n')
            logger_output.flush()

            print(printlog+'\n\n')

    printlog = ''

    try:
        avg_test = test_epoch / j
        avg_val = val_epoch / j

        if avg_val < best_val_epoch:
            best_val_epoch = avg_val
            best_test_epoch = avg_test

        printlog = printlog + '\n' + '| Test: {} | Optimum: {} | Val: {} | Best Val: {} |'.format(str(avg_test), str(best_test_epoch), str(avg_val), str(best_val_epoch))

        logger_epoch_output.write(printlog+'\n\n')
        logger_epoch_output.flush()
        print(printlog+'\n\n')
    except:
        pass

    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'best_val': best_val
    }
    torch.save(state, folder_name+'/model.pt')

for epoch in range(global_epoch, args.epochs + 1):
    train(epoch)