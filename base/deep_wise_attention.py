# coding=gbk
import torch

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def PositionalEncoding(d_model, max_seq_len):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    return pe


# class deep_Attention(torch.nn.Module):
#     def __init__(self, hidden_dim):
#         super(deep_Attention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.query_layer = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.key_layer = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.out_layer = torch.nn.Linear(hidden_dim, hidden_dim)
#
#     def forward(self, vec, f):
#
#         x = self.fc(vec)  # (batch_size, seq_len, hidden_dim)
#
#         b, seq_len, dim = x.shape
#         pe = PositionalEncoding(dim, seq_len).repeat(b, 1, 1).to(device)
#         x = x + pe
#
#         query = self.query_layer(x)
#         key = self.key_layer(x)
#
#         scores = torch.bmm(query, key.transpose(1, 2)) / self.hidden_dim ** 0.5  # (batch_size, seq_len, seq_len)
#         attention_weights = torch.softmax(scores, dim=2)  # (batch_size, seq_len, seq_len)
#
#         shape = f.shape  # b, c, d, h ,w
#         f = f.permute(0, 2, 1, 3, 4).reshape(shape[0], shape[2], -1)
#
#         context_vector = torch.bmm(attention_weights, f)
#         context_vector = self.out_layer(context_vector)
#
#         context_vector = context_vector.reshape(shape[0], shape[2], shape[1], shape[3], shape[4]).permute(0, 2, 1, 3, 4)
#
#         return context_vector

class deep_Attention(torch.nn.Module):
    def __init__(self, hidden_dim, D=32):
        super(deep_Attention, self).__init__()
        self.depth = D
        self.fc1 = torch.nn.Linear(hidden_dim, 1)
        self.fc2 = torch.nn.Linear(D, D)


    def forward(self, vec, f):

        vec = self.fc1(vec).transpose(1, 2) # B, 1, D
        vec = self.fc2(vec)
        attention_weights = torch.softmax(vec, dim=2)

        shape = f.shape  # b, c, d, h ,w
        f = f.permute(0, 2, 1, 3, 4).reshape(shape[0], shape[2], -1) # B, D, CHW

        context_vector = torch.bmm(attention_weights, f) # B, 1, CHW

        context_vector = context_vector.repeat(1, self.depth, 1)

        context_vector = context_vector.reshape(shape[0], shape[2], shape[1], shape[3], shape[4]).permute(0, 2, 1, 3, 4)

        return context_vector

if __name__ == '__main__':
    from torch import nn

    deep_fusion_32 = deep_Attention(256, 32)
    avg_pool_32 = nn.AvgPool3d(kernel_size=(1, 32, 32))

    x = torch.rand((4, 256, 32, 32, 32))
    y = avg_pool_32(x).squeeze(-1).squeeze(-1).permute(0, 2, 1)
    z = deep_fusion_32(y, x)
    print(z.shape)
