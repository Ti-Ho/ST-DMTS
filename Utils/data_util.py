import torch
from args import get_parser
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops
import warnings
import random

warnings.filterwarnings('ignore')


# load data
def load_data(dataset=None, val_split=0.0):
    """
    get data
    :param val_split:
    :param dataset: choose dataset from ["NON10", "NON12"]
    :return: data shape (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size] or None))
    """
    prefix = "datasets"
    print(f"-- loading {dataset} --")
    parser = get_parser()
    args = parser.parse_args()

    train_df = pd.read_csv(f"{prefix}/{dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"{prefix}/{dataset}/test.csv", index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    # rename the columns name
    train_df.columns = np.arange(len(train_df.columns))
    test_df.columns = np.arange(len(test_df.columns))

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)

    val_df = None
    if val_split != 0.0:
        split = int(np.floor(len(train_df.index) * (1.0 - val_split)))
        val_df = train_df[split:]
        val_df.reset_index(inplace=True, drop=True)
        train_df = train_df[:split]
        train_df.reset_index(inplace=True, drop=True)

    return train_df, val_df, test_df


# Dataset
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.tensor(data=data.values.T, dtype=torch.float32)
        self.window = window
        self.node_num, self.time_len = self.data.shape
        self.node_list = list(range(len(data.columns)))
        self.st, self.target_node = self.process()  # start_point, target_node

    def process(self):
        st_arr = np.array(list(range(0, self.time_len - self.window)) * self.node_num)  # start point
        node_arr = np.concatenate(
            ([[node] * (self.time_len - self.window) for node in self.node_list]))  # correspond target node
        return st_arr, node_arr

    def __len__(self):
        return len(self.st)

    def __getitem__(self, item):
        start_point = self.st[item]
        target_node = self.target_node[item]

        # target_data = self.data[target_node, start_point:start_point+self.window].reshape(1, -1)
        # ref_data = self.data[np.arange(self.node_num) != target_node, start_point:start_point+self.window]
        # X = torch.cat((target_data, ref_data), dim=0)
        X = self.data[:, start_point:start_point + self.window]
        # y = self.data[target_node, start_point + self.window]
        y = self.data[:, start_point + self.window]

        # return X, y, start_point, target_node
        return X, y, target_node, start_point


def build_graph(num_node):
    print("-- building graph --")
    # load node msg
    node_msg_df = pd.read_csv('datasets/node_msg.csv')

    # construct the graph
    relation_cols = ['city', 'territory_name', 'trade_area_type_fix']
    struct_map = {}
    for col in relation_cols:
        for group_df in node_msg_df.groupby(col):
            nodes = group_df[1].index
            for node_i in nodes:
                if node_i not in struct_map:
                    struct_map[node_i] = set()
                for node_j in nodes:
                    if node_j != node_i:
                        struct_map[node_i].add(node_j)

    # transform to edge list
    edge_indexes = [
        [],
        []
    ]

    for node_i, node_list in struct_map.items():
        for node_j in node_list:
            edge_indexes[0].append(node_i)
            edge_indexes[1].append(node_j)

    edge_indexes = torch.tensor(edge_indexes, dtype=torch.int32)
    edge_indexes, _ = remove_self_loops(edge_indexes)
    edge_indexes, _ = add_self_loops(edge_indexes, num_nodes=num_node)

    return edge_indexes

SEED = 42


def seed_everything(seed=SEED):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def stable(dataloader, seed=SEED):
    seed_everything(seed)
    return dataloader
