r""" Provides functions that builds/manipulates correlation tensors """
import copy
from typing import Optional
from base.position import *
import torch.nn.functional as F
from torch import nn, Tensor
from base.attention import AiAModule

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class AIAFusionNetwork(nn.Module):

    def __init__(self, d_model, batch, num_heads, depth, height, weight, feat_dim, num_featurefusion_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu", use_AiA=True):

        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, batch, num_heads, depth, height, weight, feat_dim, dim_feedforward, dropout, activation, use_AiA)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_vector, support_vector):

        B,C,D,H,W = query_vector.shape
        
        query_vector = query_vector.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
        support_vector = support_vector.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        mask = (torch.zeros(B*D, H, W) > 0.5).to(device)

        x = NestedTensor(tensors=query_vector, mask=mask)
        pos_emb = PositionEmbeddingSine(C // 2, normalize=True)
        pos_encoding = pos_emb(x)

        query_vector = query_vector.flatten(2).permute(2, 0, 1)
        support_vector = support_vector.flatten(2).permute(2, 0, 1)
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)

        output = self.encoder(src1=query_vector, src2=support_vector, pos=pos_encoding)
        output = output.permute(1, 2, 0).reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

        return output


class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, batch, num_heads, depth, height, weight, feat_dim, dim_feedforward=2048, dropout=0.1,
                 activation="relu", use_AiA=True):
        super().__init__()

        self.cross_attn = AiAModule(d_model, batch, num_heads, depth, height, weight, feat_dim, dropout=dropout, use_AiA=use_AiA)
        # self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     pos: Optional[Tensor] = None,
                     ):

        src1 = src1.to(device)
        src2 = src2.to(device)
        pos = pos.to(device)

        q = self.with_pos_embed(src1, pos)
        k = self.with_pos_embed(src2, pos)

        src12, _ = self.cross_attn(query=q, key=k, value=src2)
        src1 = src1 + self.dropout1(src12)
        src1 = self.norm1(src1)

        src12 = self.linear2(self.dropout2(self.activation1(self.linear1(src1))))
        src1 = src1 + self.dropout3(src12)
        src1 = self.norm3(src1)

        return src1

    def forward(self, src1, src2,
                pos: Optional[Tensor] = None,):

        return self.forward_post(src1, src2, pos)


class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                pos: Optional[Tensor] = None
                ):
        output1 = src1
        output2 = src2
        output = None

        for layer in self.layers:
            output = layer(output1, output2, pos)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_corr_network(hidden_dim, dropout, batch, nheads, depth, height, weight, feat_dim, dim_feedforward, featurefusion_layers, activation, use_AiA):
    return AIAFusionNetwork(
        d_model=hidden_dim,
        dropout=dropout,
        batch=batch,
        num_heads=nheads,
        depth=depth,
        height=height,
        weight=weight,
        feat_dim=feat_dim,
        dim_feedforward=dim_feedforward,
        num_featurefusion_layers=featurefusion_layers,
        activation=activation,
        use_AiA=use_AiA
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")