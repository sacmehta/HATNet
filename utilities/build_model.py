# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

from utilities.print_utilities import *
from utilities import supported_datasets


def build_model(opts, diag_classes, base_feature_odim):
    '''
    This function is to load the Medical Imaging Model

    :param opts: Arguments
    :param diag_classes: Number of diagnostic classes
    :param base_feature_odim: Output dimension of base feature extractor such as CNN
    :return:
    '''
    mi_model = None
    if opts.dataset in supported_datasets:
        from model.mi_model_e2e import MIModel
        assert opts.bag_size % opts.word_size == 0, 'Bag size should be divisible by word size'

        num_bags_words = opts.bag_size//opts.word_size
        mi_model = MIModel(n_classes=diag_classes,
                           cnn_feature_sz=base_feature_odim,
                           out_features=opts.out_features,
                           num_bags_words=num_bags_words,
                           num_heads=opts.attn_heads,
                           dropout=opts.dropout,
                           attn_type=opts.attn_type,
                           attn_dropout=opts.attn_p,
                           attn_fn=opts.attn_fn)
    else:
        print_error_message('Model for this dataset ({}) not yet supported'.format('self.opts.dataset'))

    # sanity check to ensure that everything is fine
    if mi_model is None:
        print_error_message('Model cannot be None. Please check')

    return mi_model


def get_model_opts(parser):
    '''Model details'''
    group = parser.add_argument_group('Medical Imaging Model Details')
    group.add_argument('--out-features', type=int, default=128,
                       help='Number of output features after merging bags and words')
    group.add_argument('--checkpoint', type=str, default='', help='Checkpoint directory. If argument files exist'
                                                                  'in this directory, then arguments will be automatically'
                                                                  'loaded from that file')
    group.add_argument('--attn-heads', default=2, type=int, help='Number of attention heads')
    group.add_argument('--dropout', default=0.4, type=float, help='Dropout value')
    group.add_argument('--weights-test', default='', type=str, help='Weights file')
    group.add_argument('--max-bsz-cnn-gpu0', type=int, default=100, help='Max. batch size on GPU0')
    group.add_argument('--attn-type', type=str, default='l2', choices=['avg', 'l1', 'l2'], help='How to compute attention scores')
    group.add_argument('--attn-p', type=float, default=0.2, help='Proability to drop bag and word attention weights')
    group.add_argument('--attn-fn', type=str, default='softmax', choices=['tanh', 'sigmoid', 'softmax'],
                       help='Proability to drop bag and word attention weights')
    group.add_argument('--keep-best-k-models', default=-1, type=int, help='Number of best checkpoints to be saved')

    return parser
