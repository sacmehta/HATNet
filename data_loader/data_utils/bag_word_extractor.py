import openslide
from openslide import open_slide
from PIL import Image, ImageFile
import numpy as np
from transforms.img_transforms import random_transform_np, normalize_words_np
import torch
import gc
import cv2
from data_loader.data_utils.open_slide_reader import _load_image_lessthan_2_29, _load_image_morethan_2_29
import copy
from typing import Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000


'''
Code for converting image into bags and bags into words
'''

def convert_image_to_words(image_name: str,
                           bag_width: Optional[int]=1024,
                           bag_height: Optional[int]=1024,
                           num_bags_h: Optional[int]=4,
                           num_bags_w: Optional[int]=4,
                           word_width: Optional[int]=256,
                           word_height: Optional[int]=256,
                           is_training: Optional[bool]=False,
                           return_orig_wsi_np: Optional[int]=False):
    '''
    :param image_name: Name of the image (e.g. test.tiff)
    :param bag_width: Width of the bag
    :param bag_height: Height of the bag
    :param num_bags_h: Number of bags along height dimension
    :param num_bags_w: NUmber of bags along width dimension
    :param word_width: Word width
    :param word_height: Word height
    :param is_training: Is training or evaluation
    :param return_orig_wsi_np: If True, the function will return the original image in numpy
    :return:
    '''

    assert bag_height % word_height == 0, 'Bag height ({}) should be divisible by word height ({})'.format(
        bag_height, word_height)
    assert bag_width % word_width == 0, 'Bag width ({}) should be divisible by word width ({})'.format(bag_width,
                                                                                                       word_width)

    try:
        wsi_os = open_slide(image_name)
        #    number of levels in the slide
        # levels = wsi_os.level_count
        # assert levels == 1

        width, height = wsi_os.dimensions

        if (width * height) >= 2 ** 29:
            openslide.lowlevel._load_image = _load_image_morethan_2_29
        else:
            openslide.lowlevel._load_image = _load_image_lessthan_2_29

        wsi = wsi_os.read_region((0, 0), 0, (width, height))
        wsi = wsi.convert('RGB')
    except:
        wsi = Image.open(image_name).convert('RGB')

    # convert to numpy
    wsi = np.array(wsi)

    if wsi.shape[2] > 3:
        wsi = wsi[:, :, :-1]  # discard the alpha channel

    num_bags = num_bags_h * num_bags_w

    wsi = cv2.resize(wsi, (bag_width * num_bags_w, bag_height * num_bags_h))
    if return_orig_wsi_np:
        wsi_copy = copy.deepcopy(wsi)

    # apply data transforms to the wsi during training
    if is_training:
        wsi = random_transform_np(wsi)

    roi_h, roi_w, channel = wsi.shape[:3]

    ## Notations
    # H--> Height, W --> Width , C --> Channels,
    # N_B_w, N_B_h --> number of bags along width and height, respectively
    # B_W, B_H --> Width and height of the bags, respectively
    # W_w, W_h --> Width and height of words inside each bag, respectively
    # N_W_w, N_W_h --> Number of words along width and height, respectively

    ## Image to BAGS
    # [HxWxC] --> [H x N_B_w x B_W x C]
    bags = np.reshape(wsi, (roi_h, num_bags_h, bag_width, channel))
    # [H x N_B_w x B_W x C] --> [N_B_w x H x B_W x C]
    bags = bags.transpose(1, 0, 2, 3)
    # [N_B_w x H x B_W x C] --> [N_B_w * N_B_h x B_H x B_W x C]
    bags = np.reshape(bags, (num_bags, bag_height, bag_width, channel))


    #### BAGS TO WORDS
    # N_B x B_H x B_W x C --> N_B x B_H x N_W_w x W_w x C
    words = np.reshape(bags, (num_bags, bag_height, -1, word_width, channel))
    # N_B x B_H x N_W_w x W_w x C --> N_B x N_W_w x B_H x W_w x C
    words = words.transpose(0, 2, 1, 3, 4)
    # N_B x N_W_w x B_H x W_h x C --> N_B x (N_W_w * N_W_h) x W_h x W_w x C
    words = np.reshape(words, (num_bags, -1, word_height, word_width, channel))

    # normalize the words
    words = normalize_words_np(words)

    # convert from NUMPY to TORCH
    words_torch = torch.from_numpy(words).float()

    del words

    # call the garbage collector
    gc.collect()

    if return_orig_wsi_np:
        return words_torch, wsi_copy
    else:
        return words_torch
