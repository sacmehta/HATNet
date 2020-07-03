import torch
from torch import nn
from nn_layers.ffn import FFN
from nn_layers.attn_layers import SelfAttention, ContextualAttention
from typing import NamedTuple, Optional
from torch import Tensor

AttentionScores = NamedTuple(
    "AttentionScores",
    [
        ("w2w_self_attn", Tensor),
        ("b2b_self_attn", Tensor),  # B x T
        ("word_scores", Tensor),  # B x T x C
        ("bag_scores", Tensor),  # List[T x B x C]
    ],
)


class MIModel(torch.nn.Module):
    '''
        Hollistic Attention Network
    '''
    def __init__(self, n_classes, cnn_feature_sz, out_features, num_bags_words,
                 num_heads=2, dropout=0.4, attn_type='l2', attn_dropout=0.2, attn_fn='tanh', *args, **kwargs):
        super(MIModel, self).__init__()

        self.project_cnn_words = nn.Linear(cnn_feature_sz, out_features)

        self.attn_ovr_words = SelfAttention(in_dim=out_features,
                                            num_heads=num_heads,
                                            p=dropout)

        self.attn_ovr_bags = ContextualAttention(in_dim=out_features,
                                                 num_heads=num_heads,
                                                 p=dropout)
        self.attn_dropout = nn.Dropout(p=attn_dropout)

        self.ffn_w2b_sa = FFN(input_dim=out_features, scale=2, p=dropout)
        self.ffn_w2b_cnn = FFN(input_dim=out_features, scale=2, p=dropout)

        self.ffn_b2s = FFN(input_dim=out_features, scale=2, p=dropout)

        self.bag_word_wt = nn.Linear(num_bags_words**2, num_bags_words**2, bias=False)

        self.classifier = nn.Linear(out_features, n_classes)

        self.attn_fn = None
        if attn_fn == 'softmax':
            self.attn_fn = nn.Softmax(dim=-1)
        elif attn_fn == 'sigmoid':
            self.attn_fn = nn.Sigmoid()
        elif attn_fn == 'tanh':
            self.attn_fn = nn.Tanh()
        else:
            raise ValueError('Attention function = {} not yet supported'.format(attn_fn))

        self.attn_type = attn_type

        self.reset_params()

    def reset_params(self):
        '''
        Function to initialze the parameters
        '''
        from torch.nn import init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def energy_function(self, x, need_attn=False):
        N = x.size(-1)
        if self.attn_type == 'l1':
            # absolute values
            x = torch.norm(x, p=1, dim=-1)
        elif self.attn_type == 'l2':
            # square before average
            x = torch.norm(x, p=2, dim=-1)
        else:
            # using simple average one
            x = torch.sum(x, dim=-1)
        # divide by vector length
        x = torch.div(x, N)
        energy: Tensor[Optional] = None
        if need_attn:
            energy = x

        x = self.bag_word_wt(x)
        x = self.attn_fn(x).unsqueeze(dim=-2)
        return self.attn_dropout(x), energy

    def forward(self, words, *args, **kwargs):
        '''
        :param words: Tensor of shape (B x N_b x N_w x F), where F is CNN dimension
        :param need_attn: boolean indicating if attention weights are required or not
        :return: A B x C_d vector, where C_d is the number of diagnostic classes
        '''

        need_attn = kwargs.get('need_attn', False)

        # STEP 1: Project CNN encoded words
        # (B x N_b x N_w x F) --> (B x N_b x N_w x d)
        words_cnn = self.project_cnn_words(words)

        # STEP 2: Identify important bags using self attention
        ###### a) Self attention over words, where each word looks into every other word
        ###### b) Compute importance of bags using energy function over words
        ###### a) Combine 2a and 2b to give weighted bags

        # (B x N_b x N_w x d) --> (B x N_b x N_w x d)
        words_self_attn, w2w_attn_wts_unnorm = self.attn_ovr_words(words_cnn, need_attn=need_attn)
        # Identify important words in a bag
        words_sa_energy, words_sa_energy_unnorm = self.energy_function(words_self_attn, need_attn=need_attn)
        # Merge self-attended words with their importance scores to yeild bags
        # [B x N_b x 1 x N_w] x [B x N_b x N_w x d] --> [B x N_b x d]
        bags_from_words_self_attn = torch.matmul(words_sa_energy, words_self_attn).squeeze(-2)
        bags_from_words_self_attn = self.ffn_w2b_sa(bags_from_words_self_attn)

        # STEP 3: USE CNN words to identify important bags (Same as STEP:2)
        ###### a) Compute importance of bags using energy function over words
        ###### b) Combine 3a and Step 1 to give weighted bags

        # (B x N_b x N_w x d) --> (B x N_b x 1 x d)
        words_cnn_energy, words_cnn_energy_unnorm = self.energy_function(words_cnn, need_attn=need_attn)
        # (B x N_b x 1 x d) x (B x N_b x N_w x d) --> (B x N_b x d)
        bags_from_words_cnn = torch.matmul(words_cnn_energy, words_cnn).squeeze(-2)
        bags_from_words_cnn = self.ffn_w2b_cnn(bags_from_words_cnn)

        # STEP 4: Merge bags to yeild slide-level details
        ###### a) contextual attention over bags from STEP 2 and 3 to identify important bags
        ###### b) Compute importance of each bag using energy function
        ###### b) Combine 4a and 4b to yield slide-level details

        # (B x N_b x d) --> (B x N_b x d)
        bags_self_attn, b2b_attn_wts_unnorm = self.attn_ovr_bags(bags_from_words_cnn, bags_from_words_self_attn, need_attn=need_attn)
        # Merge bags with their importance scores to identify slide-level features
        # (B x N_b x d) --> (B x 1 x d)
        bags_energy, bags_energy_unnorm = self.energy_function(bags_self_attn, need_attn=need_attn)
        # (B x 1 x d) x (B x N_b x d) --> (B x d)
        bags_to_slide = torch.matmul(bags_energy, bags_self_attn).squeeze(-2)
        out = self.ffn_b2s(bags_to_slide)

        # STEP 5: Classify to diagnostic categories using information from Step 4
        #(B x d) --> (B x C)
        out = self.classifier(out)

        if need_attn:
            words_energy_unnorm = words_sa_energy_unnorm + words_cnn_energy_unnorm
            attn_scores = AttentionScores(
                w2w_self_attn=w2w_attn_wts_unnorm,
                b2b_self_attn=b2b_attn_wts_unnorm,
                word_scores=words_energy_unnorm,
                bag_scores=bags_energy_unnorm
            )
            return out, attn_scores
        else:
            return out
