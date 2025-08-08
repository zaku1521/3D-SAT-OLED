import os

import torch
from torch import nn
import torch.nn.functional as F

from ..transformers.models.bert.configuration_bert import BertConfig
from ..transformers.models.bert.modeling_bert import BertModel, DyT

config = BertConfig(hidden_size=512, num_attention_heads=64, num_hidden_layers=18)


class MaterialStructureEncoder(nn.Module):
    def __init__(self):
        super(MaterialStructureEncoder, self).__init__()
        self.model = BertModel(config, add_pooling_layer=False).encoder
        self.emb_layer_norm = DyT(config.hidden_size, eps=1e-05)

    def forward(self, emb, attention_mask, padding_mask):
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        def fill_attn_mask(attention_mask, padding_mask, fill_val=float("-inf")):
            if attention_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attention_mask = attention_mask.view(x.size(0), -1, seq_len, seq_len)
                attention_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attention_mask = attention_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attention_mask, padding_mask

        assert attention_mask is not None
        attention_mask, padding_mask = fill_attn_mask(attention_mask, padding_mask)
        hidden_states = self.model(hidden_states=x, attention_mask=attention_mask)
        return hidden_states


class DyTanh(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.temp = nn.Parameter(torch.tensor(1.0))

        self.temp_bias = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x_normalized = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

        dynamic_temp = F.softplus(self.temp + self.temp_bias)

        x_dynamic = torch.tanh(x_normalized / dynamic_temp)

        return x_dynamic
