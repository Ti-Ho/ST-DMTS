import torch
import torch.nn as nn

from modules import (
    NodeEmbedding,
    NodeSelection,
    FeatureGAT,
    TimeGAT,
    TransformerLayer,
    LSTMLayer,
    ForecastModel,
    ReconsModel
)


class ST_DMTS(nn.Module):
    def __init__(self,
             window_size,
             out_dim,
             temporal_embedding_dim,
             spatial_embedding_dim,
             edge_index,
             num_node,
             topK,
             use_GATv2=False,
             alpha=0.2,
             dropout=0.01,
             trans_hid_dim=64,
             trans_nhead=4,
             trans_dropout=0.01,
             lstm_hid=64,
             lstm_n_layers=1,
             lstm_dropout=0.01,
             forecast_hid_dim=32,
             forecast_n_layers=1
        ):
        super(ST_DMTS, self).__init__()
        self.edge_index = edge_index
        if edge_index is None:
            self.embedding_dim = temporal_embedding_dim
        else:
            self.embedding_dim = temporal_embedding_dim + spatial_embedding_dim
        # module
        self.node_embedding = NodeEmbedding(window_size, temporal_embedding_dim, spatial_embedding_dim, edge_index, num_node)
        self.node_selection = NodeSelection(topK)
        self.feature_gat = FeatureGAT(topK, self.embedding_dim, window_size, use_GATv2, alpha=alpha, dropout=dropout)
        self.time_gat = TimeGAT(topK, window_size, use_GATv2, alpha=alpha, dropout=dropout)
        self.transformer_layer = TransformerLayer(topK * 3, trans_hid_dim, trans_nhead, trans_dropout)
        self.lstm = LSTMLayer(topK * 3, lstm_hid, lstm_dropout, lstm_n_layers)

        self.forecast_model = ForecastModel(lstm_hid, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

    def forward(self, x, target_node):
        """
        forward propagate
        :param x: data with shape (batch_size, num_node, window_size)
        :param target_node: predict target node with shape (batch_size)
        :return:
        """
        node_embedding = self.node_embedding(x)
        x, selected_embedding = self.node_selection(x, node_embedding, target_node)  # x: (batch_size, topK, window_size); selected_embedding: (batch_size, topK, embedding_dim)

        # h_feature = self.feature_gat(x, selected_embedding, self.edge_index, selected_nodes)
        h_feature = self.feature_gat(x, selected_embedding)  # (batch_size, window_size, num_node)
        h_time = self.time_gat(x)  # (batch_size, window_size, num_node)
        h_x = x.permute(0, 2, 1)

        h = torch.cat([h_x, h_feature, h_time], dim=2)  # (batch_size, window_size, 3*num_node)
        h = self.transformer_layer(h)

        hn = self.lstm(h)
        hn = hn.view(x.shape[0], -1)

        pred_res = self.forecast_model(hn)

        return pred_res

