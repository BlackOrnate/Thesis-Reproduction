import os
import zipfile
from urllib.request import urlretrieve
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

image_size = 224
batch_size = 4096
patch_size = 16
epoch_size = 10
classes = ["pizza", "steak", "sushi"]


def download_data(source: str, destination: str):
    root_path = "../data"
    image_path = f"{root_path}/{destination}"

    if os.path.isdir(image_path):
        print(f"{image_path} directory already exists, skipping download.")
    else:
        print(f"{image_path} directory does not exist, creating it...")
        os.makedirs(image_path, exist_ok=True)

        zip_name = f"{destination}.zip"
        urlretrieve(source, f"{image_path}/{zip_name}")

        with zipfile.ZipFile(f"{image_path}/{zip_name}", "r") as f:
            f.extractall(image_path)

    return image_path


def create_dataloaders(image_path: str, batch_size: int, image_size: int):
    train_dir = image_path + "/train"
    test_dir = image_path + "/test"

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_dir, transform=preprocess)
    test_data = datasets.ImageFolder(test_dir, transform=preprocess)
    classes = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, classes


def show_img(img, label, classes):
    plt.imshow(img.permute(1, 2, 0))
    plt.title(classes[label])
    plt.show()


def show_linear_img(img, patch_size, label=0):
    img = img.permute(1, 2, 0)
    plt.imshow(img[:patch_size, :, :])
    plt.title(classes[label])
    plt.tight_layout()
    plt.show()


def show_patch_img(img, patch_size, label=0):
    img = img.permute(1, 2, 0)

    num_patch = image_size // patch_size

    fig, axes = plt.subplots(1, num_patch, figsize=(num_patch * 2, 2), sharex=True, sharey=True)

    for i, ax in enumerate(axes):
        ax.imshow(img[:patch_size, patch_size * i: patch_size * (i + 1), :], cmap='viridis')
        ax.set_title(f"#{i + 1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def reshape_img(img, patch_size, label=0):
    C, H, W = img.shape[0], img.shape[1], img.shape[2]  # [3, 224, 224]
    P = patch_size
    N = (H // P) * (W // P)

    # show_linear_img(img, patch_size, label)

    # show_patch_img(img, patch_size, label)

    img = img.view(N, P ** 2 * C)  # [196, 768]
    return img
