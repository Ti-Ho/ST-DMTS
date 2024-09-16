import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LSTM
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from random import choice


class NodeEmbedding(nn.Module):
    def __init__(self, window_size, temporal_dim, spatial_dim, edge_index, num_node, tem_alpha=0.9, device="cuda"):
        super(NodeEmbedding, self).__init__()
        self.spatial_embedding_res = None
        self.window_size = window_size
        # temporal parameters
        self.temporal_dim = temporal_dim
        self.tem_alpha = tem_alpha
        # temporal embedding
        self.temporal_fc = nn.Linear(window_size, temporal_dim)
        self.temporal_embedding = torch.randn((num_node, temporal_dim), requires_grad=True,
                                              dtype=torch.float32, device=device)
        self.batch_temporal_embedding = None

        # spatial parameters
        self.spatial_dim = spatial_dim
        self.num_node = num_node
        self.edge_index = edge_index.to(device) if edge_index is not None else None
        # spatial embedding
        self.spatial_embedding = torch.randn((num_node, spatial_dim), requires_grad=True,
                                             dtype=torch.float32, device=device)
        self.batch_spatial_embedding = None
        self.conv = GCNConv(spatial_dim, spatial_dim)

    def forward(self, x):
        # temporal embedding
        # if self.temporal_embedding is None:
        #     self.batch_temporal_embedding = self.temporal_fc(x)
        # else:
        #     self.batch_temporal_embedding = self.tem_alpha * self.temporal_embedding.repeat(x.size(0), 1, 1).view(-1, self.num_node, self.temporal_dim) \
        #                                     + (1 - self.tem_alpha) * self.temporal_fc(x)

        self.batch_temporal_embedding = \
            self.tem_alpha * self.temporal_embedding.repeat(x.size(0), 1, 1).view(-1, self.num_node, self.temporal_dim) \
            + (1 - self.tem_alpha) * self.temporal_fc(x)
        # self.temporal_embedding = torch.mean(self.batch_temporal_embedding, dim=0)

        # spatial embedding
        if self.edge_index is not None:
            self.spatial_embedding_res = self.conv(self.spatial_embedding, self.edge_index)
            self.batch_spatial_embedding = self.spatial_embedding_res.repeat(x.size(0), 1, 1).view(-1, self.num_node,
                                                                                                   self.spatial_dim)
            return torch.cat((self.batch_temporal_embedding, self.batch_spatial_embedding), dim=2)
        else:
            return self.batch_temporal_embedding


# Select Most similar nodes by node embedding
class NodeSelection(nn.Module):
    def __init__(self, TopK):
        # super(NodeSelection).__init__()
        super().__init__()
        self.TopK = TopK

    def forward(self, x, node_embedding, target_node):
        """
        Choose TopK the most relevant to target node by node embedding
        :param node_embedding:
        :param x:
        :param target_node: target predict node
        :return:
            x: select topk data from x
            selected_embedding: select topk embedding from node_embeddings
        """
        N = x.size(0)
        embeddings = node_embedding.detach().clone()
        # calculate eij
        cos_mat = torch.matmul(embeddings, torch.permute(embeddings, (0, 2, 1)))
        normed_mat = torch.matmul(embeddings.norm(dim=-1).view(N, -1, 1),
                                  embeddings.norm(dim=-1).view(N, 1, -1))
        cos_mat = cos_mat / normed_mat  # e_ij

        # calculate TopK
        _, topK_indices = torch.topk(cos_mat, self.TopK, dim=-1)  # shape: (batch_size, num_node, TopK)
        targets = target_node.detach().clone().tolist()  # shape: (batch_size,)
        target_topK_indices = topK_indices[range(N), targets, :]  # (batch_size, topK)

        # # ablation_study - 'random selection nodes'
        # target_ref = []
        # for n_i in range(N):
        #     tmp = [targets[n_i]]
        #     for j in range(self.TopK - 1):
        #         tmp.append(choice([i for i in range(0, 500) if i != targets[n_i]]))
        #     target_ref.append(tmp)
        # target_topK_indices = torch.tensor(target_ref)

        # get topK input data
        x = x[np.arange(N).reshape(-1, 1), target_topK_indices.tolist(), :]
        # get topK embeddings
        node_embedding = node_embedding[np.arange(N).reshape(-1, 1), target_topK_indices.tolist(), :]

        # return x, node_embedding, target_topK_indices
        return x, node_embedding


# Feature-oriented Attention Layer
# ToDo 结合图结构有问题 可以之后再尝试 AssertionError: Static graphs not supported in 'GATConv'
# class FeatureGAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, use_v2):
#         super(FeatureGAT, self).__init__()
#         self.use_v2 = use_v2
#         self.conv = GATConv(in_dim, hidden_dim)
#         self.conv_v2 = GATv2Conv(in_dim, hidden_dim)
#         # self.conv = GCNConv(is(in_dim, hidden_dim)
#
#     def forward(self, x, node_embedding, edge_index, node_subset):
#         # edge_index = subgraph(subset=node_subset, edge_index=edge_index)[0]
#
#         x_emb = torch.concat([x, node_embedding], dim=2)
#
#         if self.use_v2:
#             x_emb = self.conv_v2(x_emb, edge_index)
#         else:
#             x_emb = self.conv(x_emb, edge_index)
#
#         x_emb = F.relu(x_emb)
#         x_emb = F.dropout(x_emb, training=self.training)
#
#         return x_emb
class FeatureGAT(nn.Module):
    def __init__(self,
                 n_features,
                 embedding_dim,
                 window_size,
                 use_gatv2,
                 hidden_dim=None,
                 use_bias=True,
                 alpha=0.2,
                 dropout=0.2):
        super(FeatureGAT, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else window_size
        self.use_bias = use_bias
        self.use_gatv2 = use_gatv2
        self.dropout = dropout
        self.x_embed_dim = embedding_dim + window_size

        if use_gatv2:
            self.hidden_dim *= 2
            linear_input_dim = self.x_embed_dim * 2
            a_input_dim = self.hidden_dim
        else:
            linear_input_dim = self.x_embed_dim
            a_input_dim = self.hidden_dim * 2

        self.linear = nn.Linear(linear_input_dim, self.hidden_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, embedding):
        # x shape: (batch_size, num_node, window_size)
        # embedding shape: (batch_size, num_node, embedding_size)
        x_emb = torch.concat([x, embedding], dim=2)
        if self.use_gatv2:
            a_input = self._make_attention_input(x_emb)
            a_input = self.leakyrelu(self.linear(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            Wx = self.linear(x_emb)
            a_input = self._make_attention_input(Wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (batch_size, num_node, num_node)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)  # (batch_size, window_size, num_node)

    def _make_attention_input(self, v):
        K = self.n_features
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.x_embed_dim)
        else:
            return combined.view(v.size(0), K, K, 2 * self.hidden_dim)


class TimeGAT(nn.Module):
    def __init__(self,
                 n_features,
                 window_size,
                 use_gatv2,
                 hidden_dim=None,
                 use_bias=True,
                 alpha=0.2,
                 dropout=0.2):
        super(TimeGAT, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else window_size
        self.use_bias = use_bias
        self.use_gatv2 = use_gatv2
        self.dropout = dropout

        if use_gatv2:
            self.hidden_dim *= 2
            linear_input_dim = self.n_features * 2
            a_input_dim = self.hidden_dim
        else:
            linear_input_dim = self.n_features
            a_input_dim = self.hidden_dim * 2

        self.linear = nn.Linear(linear_input_dim, self.hidden_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_node, window_size)
        # embedding shape: (batch_size, num_node, embedding_size)
        x = x.permute(0, 2, 1)

        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            a_input = self.leakyrelu(self.linear(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            Wx = self.linear(x)
            a_input = self._make_attention_input(Wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (batch_size, num_node, num_node)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))

        return h  # (batch_size, window_size, num_node)

    def _make_attention_input(self, v):
        K = self.window_size
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.hidden_dim)


# Transformer Block
class TransformerLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, nhead, dropout, nlayers=1):
        super(TransformerLayer, self).__init__()
        self.encoder_layers = TransformerEncoderLayer(in_dim, nhead, hid_dim, dropout, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layers, nlayers)

    def forward(self, x):
        x = self.encoder(x)
        return x


# LSTM Layer
class LSTMLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, n_layers=1):
        super(LSTMLayer, self).__init__()
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid_dim, num_layers=n_layers,
                            batch_first=True, dropout=self.dropout)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn[-1, :, :]


# Forecast Model
class ForecastModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ForecastModel, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.LSTM(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


# Recons Model
class ReconsModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconsModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h_end = x
        # h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=0).view(x.size(0), self.window_size,
                                                                          -1)  # 这里把dim=1改为0
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

# ToDo: 使用EncDec-AD替换Reconstruction Model尝试效果
