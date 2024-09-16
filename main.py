import os
from datetime import datetime
import torch.nn as nn
from Utils.data_util import *
from model import ST_DMTS
from training import Trainer
from Utils.logger import get_logger

if __name__ == '__main__':
    seed_everything()  # fix seed
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    # -- Data Params --
    dataset = args.dataset
    window_size = args.window_size
    batch_size = args.batch_size
    val_split = args.val_split

    # -- Model Params --
    temporal_embedding_dim = args.temporal_embedding_dim
    spatial_embedding_dim = args.spatial_embedding_dim
    TopK = args.TopK

    # load data
    output_path = f'output/{dataset}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = f"{output_path}/{run_id}"

    logger = get_logger(log_path=save_path)
    logger.info(args.__dict__)

    x_train, x_val, x_test = load_data(dataset, val_split)
    num_node = len(x_train.columns)
    logger.info(f"########## Data_Std: {x_train.std().mean()} ##########")

    val_loader = None
    if x_val is not None:
        val_dataset = SlidingWindowDataset(x_val, window_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -- build graph --
    edge_index = None
    if "NON" in dataset:
        edge_index = build_graph(num_node)
    # print(edge_index.shape)

    # -- model init --
    model = ST_DMTS(
        window_size,
        args.out_dim,
        temporal_embedding_dim,
        spatial_embedding_dim,
        edge_index,
        num_node,
        TopK,
        args.GATv2
    )

    # --training stage: 1. training 2. del anomalies 3. continue training --
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fore_loss = nn.MSELoss()

    test_dataset = SlidingWindowDataset(x_test, window_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer = Trainer(
        model,
        optimizer,
        fore_loss,
        save_path,
        x_train,
        val_loader,
        test_loader,
        logger
    )

    trainer.fit()

    # -- Test Stage --
    # prepare test dataset and dataloader
    test_dataset = SlidingWindowDataset(x_test, window_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"test_size: {len(test_dataset)}")
    # evaluate on test dataset
    test_loss = trainer.evaluate(test_loader)
    logger.info(f"Test forecast loss: {test_loss[0]:.5f}")
    logger.info(f"Test rmse： {test_loss[1]:.5f}")
    logger.info(f"Test mae： {test_loss[2]:.5f}")
