import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data Params --
    parser.add_argument("--dataset", type=str.upper, default="SP500N")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)

    # -- Model Params --
    parser.add_argument("--temporal_embedding_dim", type=int, default=32)
    parser.add_argument("--spatial_embedding_dim", type=int, default=32)
    parser.add_argument("--TopK", type=int, default=64)
    parser.add_argument("--GATv2", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--out_dim", type=int, default=1)

    # -- Train Params --
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_cuda", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--print_per_epoch", type=int, default=1)

    return parser
