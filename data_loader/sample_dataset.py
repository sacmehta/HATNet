import torch
import os
from torch.utils import data
from utilities.print_utilities import *
import numpy as np


class WSIDataset(torch.utils.data.Dataset):
    '''
    This class defines the data loader for Breast biopsy WSIs
    '''
    def __init__(self, img_dir, split_file, img_extn='tiff', delimeter=','):
        '''
        :param img_dir: Location of the directory that contains WSIs
        :param split_file: Which file to use (train, val, or test)
        :param img_extn: Extension of the image (e.g. tiff) without dot
        :param delimeter: Delimeter that separates the image file name from class label
        '''
        if not os.path.isfile(split_file):
            print_error_message('{} does not exist.'.format(split_file))

        super(WSIDataset, self).__init__()
        wsi_fnames = []
        diag_labels = []
        with open(split_file, 'r') as fopen:
            lines = fopen.readlines()
            for line in lines:
                img_id, label = line.strip().split(delimeter)
                img_fname = '{}/{}.{}'.format(img_dir, img_id.strip(), img_extn)
                if not os.path.isfile(img_fname):
                    print_error_message('{} file does not exist.'.format(img_fname))
                wsi_fnames.append(img_fname)

                label = int(label.strip())
                diag_labels.append(label)

        self.wsi_fnames = wsi_fnames
        self.diag_labels = diag_labels
        self.n_classes = len(np.unique(diag_labels))

        # Uncomment and add class names here
        #self.class_names = ['Benign', 'Atypia', 'DCIS', 'Invasive']
        self.class_names = [i for i in range(self.n_classes)]

        print_info_message('Samples in {}: {}'.format(split_file, len(self.wsi_fnames)))

    def __len__(self):
        return len(self.wsi_fnames)

    def __getitem__(self, index):
        '''
        For a given index value, this function returns the name of WSI and corresponding label
        '''
        return self.wsi_fnames[index], self.diag_labels[index]
