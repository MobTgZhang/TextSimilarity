from collections import Counter
import numpy as np
import torch

def vectorize(ex, model):
    """Torchify a single example."""
    word_dict = model.word_dict
    pos_dict = model.pos_dict
    sdp_dict = model.sdp_dict
    # Index words
    seg1 = torch.LongTensor([word_dict[w] for w in ex['sentence1'][0]])
    seg2 = torch.LongTensor([word_dict[w] for w in ex['sentence2'][0]])
    pos1 = torch.LongTensor([pos_dict[w] for w in ex['sentence1'][1]])
    pos2 = torch.LongTensor([pos_dict[w] for w in ex['sentence2'][1]])
    sdp1 = torch.LongTensor([sdp_dict[w] for w in ex['sentence1'][1]])
    sdp2 = torch.LongTensor([sdp_dict[w] for w in ex['sentence2'][1]])
    # Index labels
    targets = torch.FloatTensor([int(v) for v in ex['value']])
    document = [seg1,seg2,pos1,pos2,sdp1,sdp2]
    return document,targets

def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    seg1 = [ex[0] for ex in batch]
    seg2 = [ex[1] for ex in batch]
    pos1 = [ex[2] for ex in batch]
    pos2 = [ex[3] for ex in batch]
    sdp1 = [ex[4] for ex in batch]
    sdp2 = [ex[5] for ex in batch]
    targets = [ex[5] for ex in batch]

    targets = torch.cat(targets,dim=-2)

    # Batch documents and features
    len1 = max([d.size(0) for d in seg1])
    len2 = max([d.size(0) for d in seg2])
    max_length = max(len1,len2)
    batch_size = len(batch)
    segtensor1 = torch.LongTensor(batch_size,max_length)
    segtensor2 = torch.LongTensor(batch_size,max_length)
    postensor1 = torch.LongTensor(batch_size,max_length)
    postensor2 = torch.LongTensor(batch_size,max_length)
    sdptensor1 = torch.LongTensor(batch_size,max_length)
    sdptensor2 = torch.LongTensor(batch_size,max_length)
    for i in range(batch_size):
        d = seg1[i]
        segtensor1[i, :d.size(0)].copy_(d)
        d = seg2[i]
        segtensor2[i, :d.size(0)].copy_(d)
        d = pos1[i]
        postensor1[i, :d.size(0)].copy_(d)
        d = pos2[i]
        postensor2[i, :d.size(0)].copy_(d)
        d = sdp1[i]
        sdptensor1[i, :d.size(0)].copy_(d)
        d = sdp2[i]
        sdptensor2[i, :d.size(0)].copy_(d)
    return segtensor1,segtensor2,postensor1,postensor2,sdptensor1,sdptensor2,targets
