# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================
import torch
from config.opts import get_config
from train_and_eval.trainer import Trainer
from utilities.utils import (
    save_arguments,
    load_arguments
)
import os
from utilities.print_utilities import *
import json

if __name__ == '__main__':
    # get argumetns
    opts, parser = get_config()

    torch.set_default_dtype(torch.float32)

    argument_fname='mimodel_{}_bag_{}_word_{}_{}_{}'.format('config',
                                                                    opts.bag_size,
                                                                    opts.word_size,
                                                                    opts.attn_fn,
                                                                    opts.attn_type,
                                                                    )

    if not opts.checkpoint:
        # dump the arguments
        if not os.path.isdir(opts.savedir):
            os.makedirs(opts.savedir)
        save_arguments(args=opts, save_loc=opts.savedir, json_file_name=argument_fname)
        print_log_message('Config file saved here: {}'.format(opts.checkpoint))
    else:
        opts = load_arguments(parser=parser, dumped_arg_loc=opts.checkpoint, json_file_name=argument_fname)
        print_log_message('Config file loaded from {}'.format(opts.checkpoint))

    print_log_message('Arguments')
    print(json.dumps(vars(opts), indent=4, sort_keys=True))

    # run the training code
    trainer = Trainer(opts=opts)
    trainer.run()
