# python: 3.5
# encoding: utf-8
# time: 18/11/09
# author: zhen ke
# FastNLP: v_181107

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, args):
        super(TextCNN, self).__init__()

        vocab_size = args["vocab_size"]
        pretrained_embed = args["pretrained_embed"]
        padding_idx = args["padding_idx"]
        num_classes = 1
        kernel_nums = [100, 100, 100]
        kernel_sizes = [3, 4, 5]
        embed_dim = 300
        hidden_dim = 100
        drop_prob = 0.5

        # no support for pre-trained embedding currently
        if pretrained_embed is None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embed, freeze=False)
        self.embed.padding_idx = padding_idx
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, kn, ks)
             for kn, ks in zip(kernel_nums, kernel_sizes)])
        self.fc = nn.Linear(sum(kernel_nums), hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_dim, num_classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, word_seq):
        # embed
        e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]

        # conv and pool, [b,msl,e]->[b,h,msl]
        e = e.transpose(1, 2)  # [b,msl,e]->[b,e,msl]
        ps = []
        for conv in self.convs:
            c = conv(e)  # [b,e,msl]->[b,h,msl-k]
            p = F.max_pool1d(c, kernel_size=c.size(-1)).squeeze(-1)  # [b,h]
            ps.append(p)
        p = torch.cat(ps, dim=1)  # [b,h]

        # feed-forward, [b,h]->[b]
        f = self.drop(self.act(self.fc(p)))
        logits = self.out(f).squeeze(-1)

        return logits
