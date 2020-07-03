# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

from utilities.print_utilities import *
import os
import torch
import argparse
import json
import glob
from utilities.save_dict_to_file import DictWriter


def save_checkpoint(epoch, model_state, optimizer_state, best_perf, save_dir, is_best, keep_best_k_models=-1):
    best_perf = round(best_perf, 3)
    checkpoint = {
        'epoch': epoch,
        'state_dict': model_state,
        'optim_dict': optimizer_state,
        'best_perf': best_perf
    }
    # overwrite last checkpoint everytime
    ckpt_fname = '{}/checkpoint_last.pth'.format(save_dir)
    torch.save(checkpoint, ckpt_fname)

    # write checkpoint for every epoch
    ep_ckpt_fname = '{}/model_{}.pth'.format(save_dir, epoch)
    torch.save(checkpoint['state_dict'], ep_ckpt_fname)

    if keep_best_k_models > 0:
        checkpoint_files = glob.glob('{}/model_best_*')
        n_best_chkpts = len(checkpoint_files)
        if n_best_chkpts >= keep_best_k_models:
            # Extract accuracy of existing best checkpoints
            perf_tie = dict()
            for f_name in checkpoint_files:
                # first split on directory
                # second split on _
                # 3rd split on pth
                perf = float(f_name.split('/')[-1].split('_')[-1].split('.pth')[0])
                # in case multiple models have the same perf value
                if perf not in perf_tie:
                    perf_tie[perf] = [f_name]
                else:
                    perf_tie[perf].append(f_name)

            min_perf_k_checks = min(list(perf_tie.keys()))

            if best_perf >= min_perf_k_checks:
                best_ckpt_fname = '{}/model_best_{}_{}.pth'.format(save_dir, epoch, best_perf)
                torch.save(checkpoint['state_dict'], best_ckpt_fname)

                min_check_loc = perf_tie[min_acc][0]
                if os.path.isfile(min_check_loc):
                    os.remove(min_check_loc)
        else:
            best_ckpt_fname = '{}/model_best_{}_{}.pth'.format(save_dir, epoch, best_perf)
            torch.save(checkpoint['state_dict'], best_ckpt_fname)

    # save the best checkpoint
    if is_best:
        best_model_fname = '{}/model_best.pth'.format(save_dir)
        torch.save(model_state, best_model_fname)

    print_info_message('Checkpoint saved at: {}'.format(ep_ckpt_fname))


def load_checkpoint(checkpoint_dir, device='cpu'):
    ckpt_fname = '{}/checkpoint_last.pth'.format(checkpoint_dir)
    checkpoint = torch.load(ckpt_fname, map_location=device)

    epoch = checkpoint['epoch']
    model_state = checkpoint['state_dict']
    optim_state = checkpoint['optim_dict']
    best_perf = checkpoint['best_perf']
    return epoch, model_state, optim_state, best_perf

def save_arguments(args, save_loc, json_file_name='arguments.json'):
    argparse_dict = vars(args)
    arg_fname = '{}/{}'.format(save_loc, json_file_name)
    writer = DictWriter(file_name=arg_fname, format='json')
    writer.write(argparse_dict)
    print_log_message('Arguments are dumped here: {}'.format(arg_fname))


def load_arguments(parser, dumped_arg_loc, json_file_name='arguments.json'):
    arg_fname = '{}/{}'.format(dumped_arg_loc, json_file_name)
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)

        updated_args = parser.parse_args()

    return updated_args


def load_arguments_file(parser, arg_fname):
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)
        updated_args = parser.parse_args()

    return updated_args

if __name__ == '__main__':
    curr_acc = 2
    keep_best_k_models=10
    checkpoint_files = ['a_1_5.6.pth', 'a_2_0.4.pth', 'a_3_3.0.pth', 'a_4_0.4.pth']
    checkpoints_perf = dict()
    perf_tie = dict()
    for f_name in checkpoint_files:
        # first split on directory
        # second split on _
        # 3rd split on pth
        perf = float(f_name.split('/')[-1].split('_')[-1].split('.pth')[0])
        checkpoints_perf[f_name] = perf
        if perf not in perf_tie:
            perf_tie[perf] = [f_name]
        else:
            perf_tie[perf].append(f_name)

    min_acc = min(list(perf_tie.keys()))
    print(perf_tie)
    if curr_acc >= min_acc:
        check_to_delete = perf_tie[min_acc][0]
        print(check_to_delete)
    exit()

    print(checkpoints_perf)
    # sort by value
    sorted_checkpoints_perf = [(k, v) for k, v in sorted(checkpoints_perf.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True
                                                    )]
    sorted_checkpoints_perf = sorted_checkpoints_perf[:keep_best_k_models]
    min_perf_k_checks = sorted_checkpoints_perf[-1][0]

    print(min_perf_k_checks)
