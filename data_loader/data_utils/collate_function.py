import torch
import random
from data_loader.data_utils.bag_word_extractor import convert_image_to_words
import math
from typing import Optional


def get_bag_word_pairs(bag_word_size: tuple, scale_factor: int, scale_multipliers: list):
    '''
    This function returns a list of bag-word size pairs
    :param bag_word_size: Default bag-word size
    :param scale_factor: Factor by which we will increase/decrease the word size
    :param scale_multipliers: List of multipliers.
    :return: A list containing bag-word size pairs

     For example, for bag_word_size of (1024, 256), scale_factor of 32, and scale_mulitpliers of [-2, -1, 0, 1, 2],
    this function will generate bag_word pairs as:
        256 + (32 * i) for all i in [-2, -1, 0, 1, 2]
        resulting in word sizes of [192, 224, 256, 288, 320]
        We maintain the ratio between word and bag size, so our final bag_word pair list will be
        [(192*4, 192), (224*4, 224), (256*4, 256), (288*4, 288), (320*4, 320)]. The 4 comes from the ratio between
         initial bag and word size (1024/256 = 4)
    '''
    bag_sz = bag_word_size[0]
    word_sz = bag_word_size[1]
    assert bag_sz % word_sz == 0, 'Bag size should be divisible by word size. Got B: {}, W: {}'.format(bag_sz, word_sz)
    num_bags = bag_sz // word_sz
    assert num_bags >= 1, 'Number of bags should be greater than 0. Got # bags = {}'.format(num_bags)

    bag_word_pairs = []
    for m in scale_multipliers:
        word_sz_new = word_sz + (scale_factor * m)

        if word_sz_new % scale_factor != 0:
            word_sz_new = int(math.ceil(word_sz_new * 1.0 / scale_factor) * scale_factor)

        bag_sz_new = word_sz_new * num_bags

        # skip the pair if already in the list
        if (bag_sz_new, word_sz_new) in bag_word_pairs:
            continue

        bag_word_pairs.append((bag_sz_new, word_sz_new))

    return bag_word_pairs


def gen_collate_fn(batch,
                   bag_word_size: Optional[tuple]=(1024, 256),
                   is_training: Optional[bool]=False,
                   scale_factor: Optional[int]=32,
                   scale_multipliers: Optional[list]=[-2, -1, 0, 1, 2]):
    '''
    :param batch: Batch of image names and labels
    :param bag_word_size: Size of the bag and word
    :param is_training: Training or evaluation stage
    :param scale_factor: Factor by which we will increase/decrease the word size during training
    :param scale_multipliers: List of multipliers that allows to generate bags and words of different resolutions
    :return:
        Tensor of size [B x N_w x C x h_w x w_w]
        Tensor of size [B x Diag_classes]

        where   B is the batch size
                N_w is the total number of words in the image
                C is the number of channels (3 for RGB)
                h_w is the height of the word
                w_w is the width of the word
                Diag_classes is the number of diagnostic classes
    '''

    if is_training:
        # during training sample the bag-word pair randomly form a list of bag-word pairs
        bag_word_pairs = get_bag_word_pairs(bag_word_size=bag_word_size, scale_factor=scale_factor, scale_multipliers=scale_multipliers)
        bag_word_size = random.choices(bag_word_pairs)

    if isinstance(bag_word_size, list):
        bag_word_size = bag_word_size[0]

    assert bag_word_size[0] % bag_word_size[1] == 0, 'Bag size should be divisible by word size. B: {}, W: {}'.format(bag_word_size[0], bag_word_size[1])
    num_bags = bag_word_size[0] // bag_word_size[1]

    batch_words = []
    batch_labels = []
    for b_id, item in enumerate(batch):
        img_id = item[0]
        words = convert_image_to_words(image_name=img_id,
                                       bag_width=bag_word_size[0],
                                       bag_height=bag_word_size[0],
                                       num_bags_h=num_bags,
                                       num_bags_w=num_bags,
                                       word_width=bag_word_size[1],
                                       word_height=bag_word_size[1],
                                       is_training=is_training
                                       )

        # label should be a long tensor
        label = torch.LongTensor(1).fill_(item[1])

        batch_words.append(words)
        batch_labels.append(label)

    # stack words and labels
    # [B x N_w x C x h_w x w_w]
    batch_words = torch.stack(batch_words, dim=0)
    # [B x Diag_classes]
    batch_labels = torch.cat(batch_labels, dim=0)

    return batch_words, batch_labels
