from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1),
)


def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=True, train=True, root="EMNIST_data/").train_data.float()

    data_transform = Compose(
        [
            Resize([224, 224]),
            ToTensor(),
            Normalize((mnist.mean() / 255,), (mnist.std() / 255,)),
        ]
    )

    train_loader = DataLoader(
        MNIST(download=True, root="EMNIST_data/", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MNIST(
            download=False, root="EMNIST_data/", transform=data_transform, train=False
        ),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def train(model):
    # params you need to specify:
    epochs = 200

    train_loader, val_loader = get_data_loaders(64, 3)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )

    # train_loader = datasets.EMNIST(
    #     "EMNIST_data/", download=True, train=True, transform=transform, split="mnist"
    # )
    #
    loss_function = nn.CrossEntropyLoss()

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    optimizer = optim.Adadelta(model.parameters())

    losses = []
    batches = len(train_loader)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            X, y = data[0], data[1]

            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

    return model
