import torch
import numpy as np

''' 
This file calls the base feature extractor to extract word features
and then hollistic attention network to learn bag-word relationships
'''


def features_from_cnn(input_words, cnn_model, max_bsz_cnn_gpu0, num_gpus, device):
    # [B x N_b x N_w X 3 X H x W] --> [B x N_b x N_w X F]
    batch_size, num_bags, num_words, word_channels, word_height, word_height = input_words.size()
    input_words = input_words.contiguous().view(-1, word_channels, word_height, word_height)
    with torch.no_grad():
        b_sz = input_words.size(0)
        indexes = np.arange(0, b_sz, max_bsz_cnn_gpu0 * num_gpus) if num_gpus > 0 else np.arange(0, b_sz, 1)
        cnn_outputs = []
        for i in range(len(indexes)):
            start = indexes[i]
            if i < len(indexes) - 1:
                end = indexes[i + 1]
                batch = input_words[start:end].to(device)
            else:
                batch = input_words[start:].to(device)
            cnn_out = cnn_model(batch).cpu()
            cnn_outputs.append(cnn_out)
        cnn_outputs = torch.cat(cnn_outputs, dim=0)
        cnn_outputs = cnn_outputs.contiguous().view(batch_size, num_bags, num_words, -1)
        return cnn_outputs.detach()


def prediction(words, cnn_model, mi_model, max_bsz_cnn_gpu0, num_gpus, device, *args, **kwargs):
    word_features = features_from_cnn(input_words=words,
                                      cnn_model=cnn_model,
                                      max_bsz_cnn_gpu0=max_bsz_cnn_gpu0,
                                      num_gpus=num_gpus,
                                      device=device)

    word_features = word_features.to(device=device)

    # [B x N_b x N_w X F] --> [B x classes] + optional attention outputs
    output = mi_model(word_features, *args, **kwargs)

    return output