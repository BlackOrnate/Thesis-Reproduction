import torch
from torch import optim
from tools import download_data, create_dataloaders, show_img, reshape_img
from ViT import Net
from torchinfo import summary

image_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi")

image_size = 224
batch_size = 4096
patch_size = 16
epoch_size = 300
classes = ["pizza", "steak", "sushi"]


def train_loop(epoch_size, model, dataloader, optimizer=None, loss_fn=None):
    for epoch in range(1, epoch_size + 1):
        total_loss = 0.0
        for imgs, labels in dataloader:
            # show_img(imgs[0], labels[0], classes)

            imgs = imgs.to(device)  # [75, 3, 224, 224]
            labels = labels.to(device)  # [75]
            # img = reshape_img(img, patch_size, label)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader, classes = create_dataloaders(image_path=image_path, batch_size=batch_size,
                                                                image_size=image_size)

model = Net(image_size=image_size, patch_size=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3)
loss_fn = torch.nn.CrossEntropyLoss()

summary(model, input_size=(75, 3, image_size, image_size))

train_loop(epoch_size, model, test_dataloader, optimizer, loss_fn)
