import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv4d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 feat_size,
                 feat_dim,
                 bias=True,
                 transposed_query=False,
                 transposed_supp=False,
                 output_padding=None
                 ):
        super().__init__()

        if transposed_query:
            assert output_padding is not None, 'output_padding cannot be None'
            self.query_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                                                 bias=bias, padding=padding[:2], output_padding=output_padding[:2])
        else:
            self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                                        bias=bias, padding=padding[:2])

        if transposed_supp:
            assert output_padding is not None, 'output_padding cannot be None'
            self.supp_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                                                bias=bias, padding=padding[2:], output_padding=output_padding[2:])
        else:
            self.supp_conv = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                                       bias=bias, padding=padding[2:])

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.change_supp = False
        self.change_query = False

        self.linear_q1 = nn.Linear(feat_size, feat_dim)
        self.linear_q2 = nn.Linear(feat_dim, feat_size)

        self.linear_s1 = nn.Linear(feat_size, feat_dim)
        self.linear_s2 = nn.Linear(feat_dim, feat_size)

    def forward(self, x):
        """
        x: Hyper correlation map.
            shape: B L H_q W_q H_s W_s
        """
        B, L, D, H_q, W_q, H_s, W_s = x.size()

        if self.change_supp:
            x_query = self.pool_supp(x)
            H_s, W_s = x_query.shape[-2:]
        else:
            x_query = x.clone()

        if self.change_query:
            x_supp = self.pool_query(x)
            H_q, W_q = x_supp.shape[2:4]
        else:
            x_supp = x.clone()

        x_query = rearrange(x_query, 'B L D H_q W_q H_s W_s -> (B H_s W_s) L D (H_q W_q)')
        x_query = self.linear_q1(x_query)
        x_query = self.query_conv(x_query)
        x_query = self.linear_q2(x_query)
        x_query = rearrange(x_query, '(B H_s W_s) L D (H_q W_q) -> B L D H_q W_q H_s W_s', H_q=H_q, W_q=W_q, H_s=H_s,
                            W_s=W_s)

        x_supp = rearrange(x_supp, 'B L D H_q W_q H_s W_s -> (B H_q W_q) L D (H_s W_s)')
        x_supp = self.linear_s1(x_supp)
        x_supp = self.supp_conv(x_supp)
        x_supp = self.linear_s2(x_supp)
        x_supp = rearrange(x_supp, '(B H_q W_q) L D (H_s W_s) -> B L D H_q W_q H_s W_s', H_q=H_q, W_q=W_q, H_s=H_s,
                            W_s=W_s)

        return x_query + x_supp


class Encoder4D(nn.Module):
    def __init__(self,
                 corr_levels,
                 kernel_size,
                 stride,
                 padding,
                 group=(4,),
                 residual=True
                 ):
        super().__init__()
        self.conv4d = nn.ModuleList([])
        for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding)):
            conv4d = nn.Sequential(
                Conv4d(corr_levels[i], corr_levels[i + 1], k, s, p),
                nn.GroupNorm(group[i], corr_levels[i + 1]),
                nn.ReLU()  # No inplace for residual
            )
            self.conv4d.append(conv4d)

        self.residual = residual

    def forward(self, x):
        """
        x: Hyper correlation. B L H_q W_q H_s W_s
        """
        residuals = []
        for conv in self.conv4d:
            if self.residual:
                residuals.append(x)
            x = conv(x)
        # Patch embedding for transformer

        return x, residuals
