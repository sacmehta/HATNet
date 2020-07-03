# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

import torch
from torch import nn
from utilities.print_utilities import *
from utilities import supported_loss_fns


def build_criteria(opts, class_weights):
    '''
    Build the criterian function
    :param opts: arguments
    :return: Loss function
    '''
    criteria = None
    if opts.loss_fn == 'ce':
        if opts.label_smoothing:
            from criterions.cross_entropy import CrossEntropyWithLabelSmoothing
            criteria = CrossEntropyWithLabelSmoothing(ls_eps=opts.label_smoothing_eps)
            print_log_message('Using label smoothing value of : \n\t{}'.format(opts.label_smoothing_eps))
        else:
            criteria = nn.CrossEntropyLoss(weight=class_weights)
            class_wts_str = '\n\t'.join(['{} --> {:.3f}'.format(cl_id, class_weights[cl_id]) for cl_id in range(class_weights.size(0))])
            print_log_message('Using class-weights: \n\t{}'.format(class_wts_str))
    elif opts.loss_fn == 'bce':
        criteria = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        class_wts_str = '\n\t'.join(
            ['{} --> {:.3f}'.format(cl_id, class_weights[cl_id]) for cl_id in range(class_weights.size(0))])
        print_log_message('Using class-weights: \n\t{}'.format(class_wts_str))
    else:
        print_error_message('{} critiria not yet supported')

    # sanity check to ensure that everything is fine
    if criteria is None:
        print_error_message('Criteria function cannot be None. Please check')

    return criteria


def get_criteria_opts(parser):
    'Loss function details'
    group = parser.add_argument_group('Criteria options')
    group.add_argument('--loss-fn', default='ce', choices=supported_loss_fns,
                       help='Loss function')
    group.add_argument('--label-smoothing', action='store_true', default=False, help='Smooth labels or not')
    group.add_argument('--label-smoothing-eps', default=0.1, type=float, help='Epsilon for label smoothing')
    return parser