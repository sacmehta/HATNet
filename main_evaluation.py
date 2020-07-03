# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

from config.opts import get_config
from train_and_eval.evaluator import Evaluator
import torch
from utilities.utils import (
    save_arguments,
    load_arguments_file
)
import os
from utilities.print_utilities import *
import json


if __name__ == '__main__':
    # get argumetns
    opts, parser = get_config()

    torch.set_default_dtype(torch.float32)

    if opts.config_file:
        # load arguments fromm config file
        if not os.path.isfile(opts.config_file):
            print_error_message('Config file does not exists here: {}'.format(opts.config_file))
        opts = load_arguments_file(parser=parser, arg_fname=opts.config_file)

    print_log_message('Arguments')
    print(json.dumps(vars(opts), indent=4, sort_keys=True))

    # run the evaluation code
    evaluator = Evaluator(opts=opts)
    evaluator.run()