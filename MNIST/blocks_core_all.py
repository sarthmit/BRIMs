import torch
import torch.nn as nn

from attention import MultiHeadAttention
from BlockLSTM import BlockLSTM
from BlockGRU import BlockGRU
from sparse_grad_attn import blocked_grad
'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)
    output:
    output, hx, cx
'''

class BlocksCore(nn.Module):


    def __init__(self, ninp, nhid, num_blocks_in, num_blocks_out, topkval, step_att, do_gru, num_modules_read_input=2, use_higher=False, higher_separate_att = True):
        super(BlocksCore, self).__init__()

        if use_higher == False:
            higher_separate_att = False

        self.nhid = nhid
        if higher_separate_att:
            num_blocks_in += 1
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.ninp = ninp
        self.topkval = topkval
        self.step_att = step_att
        self.do_gru = do_gru
        if higher_separate_att:
            num_modules_read_input += 1
        self.num_modules_read_input = num_modules_read_input
        self.use_higher = use_higher
        self.higher_separate_att = higher_separate_att

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Blocks Core Initialize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("nhid: ", nhid)
        print("num_blocks_in: ", num_blocks_in)
        print("num_blocks_out: ", num_blocks_out)
        print("block_size_in: ", self.block_size_in)
        print("block_size_out: ", self.block_size_out)
        print("topkval: ", topkval)
        print("Higher separate att?", self.higher_separate_att)

        if self.use_higher:
            self.ninp = nhid + ninp
            if self.higher_separate_att:
                self.ninp = self.ninp//2

        self.mha = MultiHeadAttention(n_head=4, d_model_read=self.block_size_out, d_model_write=self.block_size_out, d_model_out=self.block_size_out, d_k=32, d_v=32, num_blocks_read=self.num_blocks_out, num_blocks_write=self.num_blocks_out, topk=self.num_blocks_out, grad_sparse=False)

        self.att_out = self.block_size_out*4 * 1

        self.inp_heads = 4
        print("Input Attention Heads: ", self.inp_heads)
        self.inp_att = MultiHeadAttention(n_head=self.inp_heads, d_model_read=self.block_size_out, d_model_write=self.ninp, d_model_out=self.att_out, d_k=64, d_v=self.att_out//self.inp_heads, num_blocks_read=num_blocks_out, num_blocks_write=num_modules_read_input,residual=False, topk=self.num_blocks_in+1, grad_sparse=False, skip_write=True)

        if do_gru:
            self.block_lstm = BlockGRU(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)
        else:
            self.block_lstm = BlockLSTM(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(self, inp, hx, cx, idx_layer, do_print=False, do_block=True):

        sz_b = inp.shape[0]

        hxl = []
        cxl = []

        inp_use = inp
        if self.use_higher:
            sz_b,nhid = inp.shape

            #print("using higher!")
            #print('inp shape', inp.shape)
            #print('hx-1 shape', hx[-1].shape)
            #inp_use = torch.cat([inp, hx[-1]], dim=1)

            inp_use = inp_use.reshape((sz_b, 1, nhid))
            hx_use = hx[-1].reshape((sz_b, 1, nhid))

            if self.higher_separate_att:
                inp_use = torch.cat([inp_use, hx_use], dim = 1)
            else:
                inp_use = torch.cat([inp_use, hx_use], dim = 2)

            #print('inp use shape', inp_use.shape)

        hx, cx = hx[idx_layer], cx[idx_layer]

        #use attention here.

        inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.ninp))

        #inp_use = inp_use.repeat(1,self.num_modules_read_input-1,1)
        #print('inp use shape before null block', inp_use.shape)
        inp_use = torch.cat([torch.zeros_like(inp_use[:,0:1,:]), inp_use], dim=1)
        #print('inp use shape after null block', inp_use.shape)

        #raise Exception('done')


        hx_reshape = hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out))


        inp_use, iatt, _ = self.inp_att(hx_reshape, inp_use, inp_use)

        iatt = iatt.reshape((self.inp_heads, sz_b, iatt.shape[1], iatt.shape[2]))
        iatt = iatt.mean(0)

        inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))


        null_score = iatt.mean((0,1))[1]

        new_mask = torch.ones_like(iatt[:,:,0])
        bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                sorted=True, largest=True,
                                k = self.num_blocks_out - self.topkval)[1]

        new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                    torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))

        mask = new_mask

        assert(torch.mean(torch.sum(mask, dim=1)).item() == self.topkval)

        mask = mask.reshape((inp_use.shape[0],self.num_blocks_out,1)).repeat((1,1,self.block_size_out)).reshape((inp_use.shape[0], self.num_blocks_out*self.block_size_out))

        mask = mask.detach()

        hx_old = hx*1.0
        cx_old = cx*1.0


        if self.do_gru:
            hx_new = self.block_lstm(inp_use, hx)
            cx_new = hx_new
        else:
            hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

        hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
        hx_new_grad_mask = blocked_grad.apply(hx_new, mask.reshape((mask.shape[0], self.num_blocks_out, self.block_size_out)))                #hx_new_grad_mask = hx_new * mask.reshape((mask.shape[0], self.num_blocks_out, self.block_size_out))
        hx_new_att,attn_out,extra_loss_att = self.mha(hx_new_grad_mask,hx_new_grad_mask,hx_new_grad_mask)
        hx_new = hx_new + hx_new_att
        hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
        extra_loss = extra_loss_att

        hx = (mask)*hx_new + (1-mask)*hx_old
        cx = (mask)*cx_new + (1-mask)*cx_old

        return hx, cx


if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)

    hx, cx = bc(inp, hx, cx)

    print('hx cx shape', hx.shape, cx.shape)
