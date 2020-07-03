import torch
from torch import optim
from utilities.print_utilities import *
from utilities import supported_optimziers


def build_optimizer(opts, model):
    '''
    Creates the optimizer
    :param opts: Arguments
    :param model: Medical imaging model.
    :return: Optimizer
    '''
    optimizer = None
    params = [p for p in model.parameters() if p.requires_grad]
    if opts.optim == 'sgd':
        print_info_message('Using SGD optimizer')
        optimizer = optim.SGD(params, lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optim == 'adam':
        print_info_message('Using ADAM optimizer')
        beta1 = 0.9 if opts.adam_beta1 is None else opts.adam_beta1
        beta2 = 0.999 if opts.adam_beta2 is None else opts.adam_beta2
        optimizer = optim.Adam(
            params,
            lr=opts.lr,
            betas=(beta1, beta2),
            weight_decay=opts.weight_decay,
            eps=1e-9)
    else:
        print_error_message('{} optimizer not yet supported'.format(opts.optim))

    # sanity check to ensure that everything is fine
    if optimizer is None:
        print_error_message('Optimizer cannot be None. Please check')

    return optimizer


def update_optimizer(optimizer, lr_value):
    '''
    Update the Learning rate in optimizer
    :param optimizer: Optimizer
    :param lr_value: Learning rate value to be used
    :return: Updated Optimizer
    '''
    optimizer.param_groups[0]['lr'] = lr_value
    return optimizer


def read_lr_from_optimzier(optimizer):
    '''
    Utility to read the current LR value of an optimizer
    :param optimizer: Optimizer
    :return: learning rate
    '''
    return optimizer.param_groups[0]['lr']


def get_optimizer_opts(parser):
    'Loss function details'
    group = parser.add_argument_group('Optimizer options')
    group.add_argument('--optim', default='sgd', type=str, choices=supported_optimziers,
                       help='Optimizer')
    group.add_argument('--adam-beta1', default=0.9, type=float, help='Beta1 for ADAM')
    group.add_argument('--adam-beta2', default=0.999,  type=float, help='Beta2 for ADAM')
    group.add_argument('--lr', default=0.0005, type=float, help='Initial learning rate for the optimizer')
    group.add_argument('--weight-decay', default=4e-6, type=float, help='Weight decay')

    group =  parser.add_argument_group('Optimizer accumulation options')
    group.add_argument('--accum-count', type=int, default=1, help='After how many iterations shall we update the weights')

    return parser

