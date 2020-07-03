import argparse
import random
import torch
import time
from utilities.build_criteria import get_criteria_opts
from utilities.build_optimizer import get_optimizer_opts
from utilities.build_model import get_model_opts
from utilities.build_dataloader import get_dataset_opts
from utilities.lr_scheduler import get_scheduler_opts
from model.base_feature_extractor import get_base_extractor_opts

'''
In this file, we define command-line arguments
'''

def general_opts(parser):
    group = parser.add_argument_group('General Options')

    group.add_argument('--log-interval', type=int, default=5, help='After how many iterations, we should print logs')
    group.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    group.add_argument('--seed', type=int, default=1882, help='Random seed')
    group.add_argument('--config-file', type=str, default='', help='Config file if exists')
    group.add_argument('--msc-eval', action='store_true', default=False, help='Multi-scale evaluation')

    return parser

def visualization_opts(parser):
    group = parser.add_argument_group('Visualization options')
    group.add_argument('--im-or-file', type=str, required=True, help='Name of the image or list of images in file to be visualized')
    group.add_argument('--is-type-file', action='store_true', default=False, help='Is it a file? ')
    group.add_argument('--img-extn-vis', type=str, required=True, help='Image extension without dot (example is png)')
    group.add_argument('--vis-res-dir', type=str, default='results_vis', help='Results after visualization')
    group.add_argument('--no-pt-files', action='store_true', default=False, help='Do not save data using torch.save')
    return parser

def get_opts(parser):
    '''General options'''
    parser = general_opts(parser)

    '''Optimzier options'''
    parser = get_optimizer_opts(parser)

    '''Loss function options'''
    parser = get_criteria_opts(parser)

    '''Medical Image model options'''
    parser = get_model_opts(parser)

    '''Dataset related options'''
    parser = get_dataset_opts(parser)

    ''' LR scheduler details'''
    parser = get_scheduler_opts(parser)

    '''Base feature extractor options'''
    parser = get_base_extractor_opts(parser)

    return parser


def get_config(is_visualization=False):
    parser = argparse.ArgumentParser(description='Medical Imaging')
    parser = get_opts(parser)
    if is_visualization:
        parser = visualization_opts(parser)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.data_workers)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.savedir = 'results_{}/{}_s_{}/sch_{}/{}/'.format(args.dataset,
                                                          args.base_extractor,
                                                          args.s,
                                                          args.scheduler,
                                                          timestr)

    return args, parser
