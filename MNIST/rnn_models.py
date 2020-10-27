import torch.nn as nn
import torch
from attention import MultiHeadAttention
from layer_conn_attention import LayerConnAttention
from BlockLSTM import BlockLSTM
import random
import time
from GroupLinearLayer import GroupLinearLayer
from sparse_grad_attn import blocked_grad

from blocks import Blocks

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, n_out=10, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=[6], topk=[4], do_gru=False,
                 use_inactive=False, blocked_grad=False, layer_dilation = -1, block_dilation = -1, num_modules_read_input=2):
        super(RNNModel, self).__init__()

        self.topk = topk
        print('Top k Blocks: ', topk)

        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)

        print('Number of Inputs, ninp: ', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        self.num_blocks = num_blocks
        self.nhid = nhid
        self.discrete_input = discrete_input
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.use_adaptive_softmax = use_adaptive_softmax

        print("Dropout rate", dropout)

        self.decoder = nn.Linear(nhid[-1], n_out)

        self.init_weights()

        self.model = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, calc_mask=False):
        extra_loss = 0.0

        emb = self.drop(self.encoder(input))

        hx, cx = hidden
        masks = []
        output = []

        self.model.blockify_params()

        for idx_step in range(input.shape[0]):
            hx, cx = self.model(emb[idx_step], hx, cx, idx_step)
            output.append(hx[-1])

        hidden = (hx,cx)
        output = torch.stack(output)
        output = self.drop(output)

        dec = output.view(output.size(0) * output.size(1), self.nhid[-1])
        dec = self.decoder(dec)

        if calc_mask:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden
        else:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden

    def init_hidden(self, bsz):
        hx, cx = [],[]
        weight = next(self.model.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))

        return (hx,cx)
