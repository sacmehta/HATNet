# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================
import numpy as np
import random
import cv2

'''
In this file, we define different transformation functions
'''

# Normalization PARAMETERS for the IMAGENET dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def normalize_words_np(words_np):
    '''
    This function normalizes words and then transforms words from
        N_B x N_W x W_h x W_w x C --> N_B x N_W x x C x W_h x W_w
    :param words_np: A numpy array of size N_B x N_W x W_h x W_w x C
    :return: transformed numpy array of size N_B x N_W x x C x W_h x W_w
    '''

    # N_B x N_W x W_h x W_w x C
    words_np = words_np.astype(float)
    # convert from [0, 255] to [0.0, 1.0]
    words_np /= 255.0

    # MEAN and STD NORMALIZATION ACROSS Last axis
    words_np -= MEAN
    words_np /= STD

    # N_B x N_W x W_h x W_w x C --> N_B x N_W x C x W_h x W_w
    words_np = words_np.transpose(0, 1, 4, 2, 3)

    return words_np


def random_transform_np(img_np, max_rotation=10):
    '''
    This function implements a transformation for np array image
    :param img_np: An RGB Image
    :param max_rotation: max rotation allowed. Angle between (-max_rotation, max_rotation) is selected randomly
    :return: transformed RGB Image
    '''
    h, w = img_np.shape[:2]
    # flip the bag
    if random.random() < 0.5:
        flip_code = random.choice([0, 1])  # 0 for horizontal and 1 for vertical
        img_np = cv2.flip(img_np, flip_code)

    # rotate the image
    if random.random() < 0.5:
        angle = random.choice(np.arange(-max_rotation, max_rotation + 1).tolist())
        # note that these functions take argument as (w, h)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h),
                                borderValue=(255, 255, 255))  # 255 because that correpond to background in WSIs

    # random crop and scale
    if random.random() < 0.5:
        x = random.randint(0, w - w // 4)
        y = random.randint(0, h - h // 4)
        img_np = img_np[y:, x:]
        img_np = cv2.resize(img_np, (w, h))

    return img_np