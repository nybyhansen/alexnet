from torch.optim.sgd import SGD
from alexnet.trainer import Trainer
from alexnet.alexnet import AlexNet
import torch.nn as nn
from torch.optim import SGD
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


def main() -> None:

    alexnet_model = AlexNet(n_classes=1_000)

    dataset_train = ImageNet(root="./imagenet", download=True, str="train")

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=4,
        num_workers=3,
    )

    optimizer = SGD(alexnet_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=alexnet_model,
        train_dataloader=dataloader_train,
        test_dataloader=None,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=1,
    )

    trainer.train()

if __name__ == "__main__":
    main()