from collections import OrderedDict

import torch

from model import CNNCTC

characters = '-aAbdeEfFgGjJknNMpPqQstTuvxyYzZ2479'


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)


def load_model(device):
    model = CNNCTC(n_classes=len(characters)).to(device)
    load_weights(model, torch.load('model.bin', map_location='cpu'))
    return model


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')
