import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm


class MyModel(nn.Module):
    def __init__(self, in_channels, out):
        super().__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, 32, (3, 3), bias=False),  # (batch_size, 3, 500, 500) => (batch_size, 32, 498, 498)
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),  # (batch_size, 32, 498, 498) => (batch_size, 32, 249, 249)

                    nn.Conv2d(32, 64, (3, 3), bias=False),  # (batch_size, 32, 249, 249) => (batch_size, 64, 247, 247)
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2)  # (batch_size, 64, 247, 247) => (batch_size, 64, 123, 123)
                )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Aдаптируем к фиксированному размеру (batch_size, 64, 1, 1)

        self.flatten = nn.Flatten()  # (batch_size, 64, 1, 1) => (batch_size, 64)

        self.fc = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, out)
                )

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class PrepDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split('/')[-1]
            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = np.array(Image.open(file_path))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Проверка на повреждения
            except (IOError, SyntaxError):
                print(f"Удаление поврежденного файла: {file_path}")
                os.remove(file_path)


def remove_alpha(image):
    if image.mode == 'RGBA':
        return image.convert("RGB")
    return image


def remove_extra_channel(tensor):
    if tensor.shape[0] == 4:  # Если есть альфа-канал
        tensor = tensor[:3, :, :]  # Удалить альфа-канал
    return tensor


if __name__ == "__main__":
    check_images('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/training/train/Cats')
    check_images('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/training/train/Cats')
    check_images('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/training/val/Dogs')
    check_images('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/training/val/Dogs')

    # Трансформации над объектом
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Lambda(remove_alpha),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((500, 500)),
            v2.Lambda(lambda x: remove_extra_channel(x) if isinstance(x, torch.Tensor) else x),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Определение параметров модели
    model = MyModel(3, 2)  # 3 входных канала, 2 класса (Cats и Dogs)
    loss_model = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                              mode='min',
                                                              factor=0.1,
                                                              patience=2,
                                                              threshold=0.0001,
                                                              threshold_mode='rel',
                                                              cooldown=0,
                                                              min_lr=0,
                                                              eps=1e-8
                                                              )

    # Перенос модели на устройство (GPU, если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = PrepDataset('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/training/', transform=transform)
    test_data = PrepDataset('D:/CNN_Cats_and_Dogs/cats_dogs_dataset/testing/', transform=transform)

    train_data, val_data = random_split(train_data, [0.7, 0.3])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    EPOCHS = 20
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_loss = None
    lr_list = []

    # Цикл обучения
    for epoch in range(EPOCHS):

        # Тренировка модели
        model.train()
        running_train_loss = []
        true_answer = 0
        train_loop = tqdm(train_loader, leave=False)
        for x, targets in train_loop:
            # x = x.reshape(-1, 64*64)
            x = x.to(device)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(2)[targets].to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}], train_loss = {mean_train_loss:.4f}")

        running_train_acc = true_answer / len(train_data)

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # Проверка модели (валидация)
        model.eval()
        with torch.no_grad():
            running_val_loss = []
            true_answer = 0
            for x, targets in val_loader:
                # x = x.reshape(-1, 64*64)
                x = x.to(device)
                targets = targets.reshape(-1).to(torch.int32)
                targets = torch.eye(2)[targets].to(device)

                pred = model(x)
                loss = loss_model(pred, targets)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        running_val_acc = true_answer / len(train_data)

        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}], train_loss = {mean_train_loss:.4f}, train_acc = {running_train_acc:.4f}, "
              f"val_loss = {mean_val_loss:.4f}, val_acc = {running_val_acc:.4f}")

        # idx = epoch
        # lr_scheduler.step(loss[idx])
        #
        # lr = lr_scheduler._last_lr[0]
        # lr_list.append(lr)

        if best_loss is None:
            best_loss = mean_val_loss

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss

            torch.save(model.state_dict(), f'model_state_dict_epoch_{epoch+1}.pt')
            print(f'На эпохе - {epoch+1} сохранена модель со значением функции потерь на валидации - '
                  f'{mean_val_loss:.4f}', end='\n\n')

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['loss_train', 'loss_val'])
    plt.show()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['acc_train', 'acc_val'])
    plt.show()

    # checkpoint = torch.load("model_state_dict_epoch19.pt")
    # model.load_state_dict(check_images(['state_model']))
    # model.eval()
    # with torch.no_grad():
    #     running_test_loss = []
    #     true_answer = 0
    #     for x, targets in test_loader:
    #         x = x.to(device)
    #         targets = targets.reshape(-1).to(torch.int32)
    #         targets = torch.eye(2)[targets].to(device)
    #
    #         pred = model(x)
    #         loss = loss_model(pred, targets)
    #
    #         running_test_loss.append(loss.item())
    #         mean_test_loss = sum(running_test_loss) / len(running_test_loss)
    #
    #         true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
    #
    #     running_test_acc = true_answer / len(train_data)
    #
    #     print(f"test_loss = {mean_test_loss:.4f}, test_acc = {running_test_acc:.4f}", end='\n\n')
