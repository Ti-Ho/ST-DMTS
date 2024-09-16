import os.path
import time
import torch.cuda
from math import sqrt
from Utils.data_util import *
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            forecast_loss,
            save_path="",
            x_train=None,
            val_loader=None,
            test_loader=None,
            logger=None
    ):
        parser = get_parser()
        args = parser.parse_args()

        # -- Data Params --
        self.val_split = args.val_split
        self.shuffle = args.shuffle_dataset
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.train_data = x_train
        self.val_loader = val_loader

        # -- Train Params --
        self.model = model
        self.optimizer = optimizer
        self.forecast_loss = forecast_loss
        self.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.n_epochs = args.epochs
        self.print_per_epoch = args.print_per_epoch
        self.lr = args.lr

        self.test_loader = test_loader

        self.losses = {
            "train_loss": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_rmse": [],
            "val_mae": []
        }

        self.logger = logger

        if self.device == "cuda":
            self.model.cuda()

    def fit(self):
        # 1. Prepare initial Dataset and DataLoader
        seed_everything()
        train_dataset = SlidingWindowDataset(self.train_data, self.window_size)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = self.val_loader

        eva_start = time.time()
        # init_train_loader = self.train_loader
        init_train_loss = self.evaluate(train_loader)
        eva_end = time.time()
        self.logger.info(f"Init total train loss: {init_train_loss[2]}, evaluating done in {eva_end - eva_start}s")

        # init_val_loader = self.val_loader
        if val_loader is not None:
            eva_start = time.time()
            init_val_loss = self.evaluate(val_loader)
            eva_end = time.time()
            self.logger.info(f"Init total val loss: {init_val_loss[2]}, evaluating done in {eva_end - eva_start}s")

        # 2. Train model
        self.logger.info(f"-- Starting Training model for {self.n_epochs} epochs --")
        train_start = time.time()
        optimal_val_rmse = None

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_train_losses = []
            # 1). Train for one epoch
            for x, y, target_node, _ in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(x, target_node)
                preds = preds.squeeze(1)
                target_y = y[np.arange(len(target_node)), target_node.tolist()]
                loss = torch.sqrt(self.forecast_loss(target_y, preds))
                loss.backward()
                self.optimizer.step()
                forecast_train_losses.append(loss.item())

            forecast_train_losses = np.array(forecast_train_losses)
            total_epoch_loss = np.sqrt((forecast_train_losses ** 2).mean())

            self.losses["train_loss"].append(total_epoch_loss)

            # 2). Evaluate on validation set
            total_val_loss, rmse, mae = None, None, None
            if val_loader is not None:
                total_val_loss, rmse, mae = self.evaluate(val_loader)
                self.losses["val_total"].append(total_val_loss)
                self.losses["val_rmse"].append(rmse)
                self.losses["val_mae"].append(mae)
                if optimal_val_rmse is None or rmse <= optimal_val_rmse:
                    optimal_val_rmse = rmse
                    self.save(f"model.pt")

            epoch_time = time.time() - epoch_start

            # print train and validation msg
            if epoch % self.print_per_epoch == 0:
                train_msg = (
                    f"[Epoch {epoch + 1}] "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    train_msg += (
                        f" ---- val_forecast_loss = {total_val_loss:.5f}, "
                        f"val_rmse = {rmse}, val_mae = {mae}, "
                        f"total_loss = {total_epoch_loss:.5f}"
                    )

                train_msg += f" [{epoch_time:.1f}s]"
                self.logger.info(train_msg)
                _, test_rmse, test_mae = self.evaluate(self.test_loader)
                self.logger.info(f"test_rmse: {test_rmse}, test_mae: {test_mae}")

        if val_loader is None:
            self.save(f"model.pt")

        # save the train record
        with open(f'{self.save_path}/train_loss_record.pkl', 'wb') as f:
            pickle.dump(self.losses, f)

        train_time = time.time() - train_start
        self.logger.info(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        self.model.eval()

        forecast_losses = []
        y_list = np.array([])
        y_hat_list = np.array([])
        with torch.no_grad():
            for x, y, target_node, _ in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x, target_node)
                preds = preds.squeeze(1)
                target_y = y[np.arange(len(target_node)), target_node.tolist()]
                forecast_loss = torch.sqrt(self.forecast_loss(target_y, preds))

                target_x = x[np.arange(len(target_node)), target_node.tolist(), :]

                forecast_losses.append(forecast_loss.item())
                y_list = np.concatenate([y_list, target_y.detach().cpu().numpy()])
                y_hat_list = np.concatenate([y_hat_list, preds.detach().cpu().numpy()])

        forecast_losses = np.array(forecast_losses)
        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        total_loss = np.sqrt((forecast_losses ** 2).mean())
        rmse = sqrt(mean_squared_error(y_list, y_hat_list))
        mae = mean_absolute_error(y_list, y_hat_list)

        return total_loss, rmse, mae

    def save(self, file_name):
        PATH = f"{self.save_path}/{file_name}"
        self.logger.info(f"-- save model to {PATH} --")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))
