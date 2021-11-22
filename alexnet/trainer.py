import torch.nn as nn
from torch.utils.data import DataLoader
# epochs: 90
# batch_size: 128
# optim: SGD
# lr: 0.01 (reduced three times when plateau. manually)
# momentum: 0.9
# weight decay: 0.0005

class Trainer:
    def __init__(self, 
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer,
        criterion,
        n_epochs: int,

    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs

    def train(self):

        for _epoch in range(self.n_epochs):
            self.model.train()

            for images_batch, labels_batch in self.train_dataloader:
                self.optimizer.zero_grad()
                model_output_batch = self.model(images_batch)
                loss = self.criterion(model_output_batch, labels_batch)

                loss.backward()
                self.optimizer.step()
