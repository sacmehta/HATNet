# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

import torch
from utilities.print_utilities import *
from utilities import supported_datasets
import numpy as np
from data_loader.data_utils.collate_function import gen_collate_fn


def get_data_loader(opts):
    '''
    Create data loaders
    :param opts: arguments
    :param base_feature_extractor: base feature extractor that transforms RGB words to vectors
    :return: train and validation dataloaders along with number of diagnostic classes
    '''
    train_loader, val_loader, diag_classes = None, None, 0
    if opts.dataset == 'bbwsi':
        from data_loader.bbwsi_dataset import BBWSIDataset
        train_dataset = BBWSIDataset(img_dir=opts.img_dir,
                                     split_file=opts.train_file,
                                     img_extn=opts.img_extn,
                                     delimeter=','
                                     )

        val_dataset = BBWSIDataset(img_dir=opts.img_dir,
                                   split_file=opts.val_file,
                                   img_extn=opts.img_extn,
                                   delimeter=','
                                   )

        diag_classes = train_dataset.n_classes
        bag_word_size = (opts.bag_size, opts.word_size)

        diag_labels = train_dataset.diag_labels

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True,
                                                   pin_memory=False,
                                                   num_workers=opts.data_workers,
                                                   collate_fn=lambda batch: gen_collate_fn(batch=batch,
                                                                                           bag_word_size=bag_word_size,
                                                                                           is_training=True,
                                                                                           scale_factor=opts.scale_factor,
                                                                                           scale_multipliers=opts.scale_multipliers
                                                                                           )
                                                   )
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opts.batch_size,
                                                 shuffle=False,
                                                 pin_memory=False,
                                                 num_workers=opts.data_workers,
                                                 collate_fn=lambda batch: gen_collate_fn(batch=batch,
                                                                                         bag_word_size=bag_word_size,
                                                                                         is_training=False
                                                                                         )
                                                 )
    else:
        print_error_message('{} dataset not supported yet'.format(opts.dataset))

    # compute class-weights for balancing dataset
    if opts.class_weights:
        # inversely propotional to class frequency
        class_weights = np.histogram(diag_labels, bins=diag_classes)[0]
        class_weights = np.array(class_weights) / sum(class_weights)

        for i in range(diag_classes):
            class_weights[i] = round(np.log(1 / class_weights[i]), 5)
    else:
        class_weights = np.ones(diag_classes, dtype=np.float)

    print_log_message('Bag size: {}, word size: {}'.format(opts.bag_size, opts.word_size))

    return train_loader, val_loader, diag_classes, class_weights


def get_test_data_loader(opts):
    '''
    Creates a data loader for test images
    :param opts: Arguments
    :param base_feature_extractor: base feature extractor that transforms RGB words to vectors
    :return: test dataloader along with number of diagnostic classes
    '''
    test_loader = None
    diag_classes = 0
    class_names = None
    if opts.dataset == 'bbwsi':
        from data_loader.bbwsi_dataset import BBWSIDataset

        test_dataset = BBWSIDataset(img_dir=opts.img_dir,
                                    split_file=opts.test_file,
                                    img_extn=opts.img_extn,
                                    delimeter=','
                                    )
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=opts.batch_size,
                                                  shuffle=False,
                                                  pin_memory=False,
                                                  num_workers=opts.data_workers,
                                                  )

        diag_classes = 4
        class_names = test_dataset.class_names
    else:
        print_error_message('{} dataset not supported yet'.format(opts.dataset))
    return test_loader, diag_classes, class_names


def get_dataset_opts(parser):
    '''
        Medical imaging Dataset details
    '''
    group = parser.add_argument_group('Dataset general details')
    group.add_argument('--img-dir', type=str, default='./data', required=True, help='Dataset location')
    group.add_argument('--img-extn', type=str, default='tiff', help='Extension of WSIs. Default is tiff')
    group.add_argument('--dataset', type=str, default='bbwsi', choices=supported_datasets, help='Dataset name')
    group.add_argument('--train-file', type=str, default='vision_datasets/breast_biopsy_wsi/train.txt',
                       help='Text file with training image ids and labels')
    group.add_argument('--val-file', type=str, default='vision_datasets/breast_biopsy_wsi/val.txt',
                       help='Text file with validation image ids and labels')
    group.add_argument('--test-file', type=str, default='vision_datasets/breast_biopsy_wsi/test.txt',
                       help='Text file with testing image ids and labels')

    group = parser.add_argument_group('Input details')
    group.add_argument('--bag-size', type=int, default=1024, help='Bag size. We use square bags')
    group.add_argument('--word-size', type=int, default=256, help='Word size. We use square bags')
    group.add_argument('--scale-factor', type=int, default=32,
                       help='Factor by which word size will be increased or decrease. '
                            'Default is 32 because ImageNet models down-sample the input image by 32')
    group.add_argument('--scale-multipliers', type=int, default=[-2, -1, 0, 1, 2], nargs="+",
                       help='Factor by which word size will be increased or decrease')

    group = parser.add_argument_group('Batching details')
    group.add_argument('--batch-size', type=int, default=1, help='Batch size')
    group.add_argument('--data-workers', type=int, default=1, help='Number of workers for data loading')

    group = parser.add_argument_group('Class-wise weights for loss fn')
    group.add_argument('--class-weights', action='store_true', default=False,
                       help='Compute normalized to address class-imbalance')

    return parser
