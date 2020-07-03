import torch
from torch import nn
from nn_layers.ffn import FFN
from nn_layers.multi_head_attn import MultiHeadAttn


'''
Adapted from OpenNMT-Py
'''

class SelfAttention(nn.Module):
    '''
    This class implements the transformer block with multi-head attention and Feed forward network
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.layer_norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, x, need_attn=False):
        '''
        :param x: Input (bags or words)
        :param need_attn: Need attention weights or not
        :return: returns the self attention output and attention weights (optional)
        '''
        x_norm = self.layer_norm_1(x)

        context, attn = self.self_attn(x_norm, x_norm, x_norm, need_attn=need_attn)

        out = self.drop(context) + x
        return self.ffn(out), attn


class ContextualAttention(torch.nn.Module):
    '''
        This class implements the contextual attention.
        For example, we used this class to compute bag-to-bag attention where
        one set of bag is directly from CNN, while the other set of bag is obtained after self-attention
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(ContextualAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.context_norm = nn.LayerNorm(in_dim)
        self.context_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)
        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, input, context, need_attn=False):
        '''
        :param input: Tensor of shape (B x N_b x N_w x CNN_DIM) or (B x N_b x CNN_DIM)
        :param context: Tensor of shape (B x N_b x N_w x hist_dim) or (B x N_b x hist_dim)
        :return:
        '''

        # Self attention on Input features
        input_norm = self.input_norm(input)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, need_attn=need_attn)
        query = self.drop(query) + input
        query_norm = self.query_norm(query)

        # Contextual attention
        context_norm = self.context_norm(context)
        mid, contextual_attn = self.context_attn(context_norm, context_norm, query_norm, need_attn= need_attn)
        output = self.ffn(self.drop(mid) + input)

        return output, contextual_attn