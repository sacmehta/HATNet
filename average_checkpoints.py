import argparse
import collections
import torch
import glob
import os
import json


def average_checkpoints(model_checkpoints):
    params_dict = collections.OrderedDict()
    params_keys = None
    new_model_params = None
    num_models = len(model_checkpoints)

    for ckpt in model_checkpoints:
        model_params = torch.load(ckpt, map_location='cpu')

        # Copies over the settings from the first checkpoint
        if new_model_params is None:
            new_model_params = model_params

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(ckpt, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    return averaged_params


def main():
    parser = argparse.ArgumentParser(description='Average checkpoints')

    parser.add_argument('--checkpoint-dir', required=True, type=str, default='results', help='Checkpoint directory location.')
    parser.add_argument('--best-n', required=True, type=int, default=5, help='Num of epochs to average')
    parser.add_argument('--after-ep', type=int, default=-1, help='Ignore first k epoch')

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    epoch_acc = json.load(open(glob.glob('{}/val*.json'.format(checkpoint_dir))[0], 'r'))
    epoch_acc = [(int(ep), acc) for ep, acc in epoch_acc.items() if int(ep) > args.after_ep]
    sorted_epoch_acc = sorted(epoch_acc, key=lambda x: (x[1], x[0]), reverse=True)

    epoch_numbers = [x[0] for x in sorted_epoch_acc[:args.best_n]]
    print(f'epochs to average: {epoch_numbers}')
    checkpoints = []

    file_names = glob.glob('{}/model_*.pth'.format(checkpoint_dir))

    for f_name in file_names:
        # first split on _ep_
        # then split on '.pth'
        if 'model' not in f_name or 'best' in f_name:
            continue

        ep_no = int(f_name.split('_')[-1].split('.pth')[0])
        if ep_no in epoch_numbers:

            if not os.path.isfile(f_name):
                print('File does not exist. {}'.format(f_name))
            else:
                checkpoints.append(f_name)

    assert len(checkpoints) > 1, 'Atleast two checkpoints are required for averaging'

    averaged_weights = average_checkpoints(checkpoints)
    ckp_name = '{}/averaged_model_best{}.pth'.format(checkpoint_dir, args.best_n) if args.after_ep == -1 else \
               '{}/averaged_model_best{}_after{}.pth'.format(checkpoint_dir, args.best_n, args.after_ep)
    torch.save(averaged_weights, ckp_name)

    print('Finished writing averaged checkpoint')


if __name__ == '__main__':
    main()