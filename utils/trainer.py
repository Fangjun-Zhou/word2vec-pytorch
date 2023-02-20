import os
import copy
import numpy as np
import json
import torch
import time
import gc


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        # Check if the path exists, if not create it.
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        
        # Use DataParallel if multiple GPUs are available.
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs.".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

    def train(self):
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model.state_dict())
        # Record the start time.
        start_time = time.time()
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            print(
                "Time elapsed: {:.2f} min, average epoch time: {:.2f} min, predicting finish time: {:.2f} min".format(
                    (time.time() - start_time) / 60,
                    (time.time() - start_time) / (epoch + 1) / 60,
                    (time.time() - start_time) / (epoch + 1) / 60 * self.epochs,
                )
            )

            self.lr_scheduler.step()
            
            if self.loss["val"][-1] < best_val_loss:
                best_val_loss = self.loss["val"][-1]
                best_model = copy.deepcopy(self.model.state_dict())

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
            
            gc.collect()
            if (torch.cuda.is_available()):
                torch.cuda.empty_cache()
        
        model_path = f"best_val_model_{best_val_loss:.2f}.pt"
        model_path = os.path.join(self.model_dir, model_path)
        torch.save(best_model, model_path)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)
    
    def load_model(self, path=None):
        """Load model from `self.model_dir` directory"""
        if (self.device.type == "cpu"):
            self.model = torch.load(path, map_location=torch.device('cpu'))
        else:
            self.model = torch.load(path)
        self.model.to(self.device)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)