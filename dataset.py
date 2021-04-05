import pickle
import torch
import torchvision.transforms as transforms
import numpy as np


class TrainDatasets(torch.utils.data.Dataset):
    def __init__(self, transform=None, mode="train"):
        self._transform = transform

        path = "data/cifar-10-batches-py/"
        if mode == "train":
            with open(path + "data_batch_1", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                batch1_data = dataset[b"data"].reshape(-1, 32, 32, 3)
                batch1_label = dataset[b"labels"]
            with open(path + "data_batch_2", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                batch2_data = dataset[b"data"].reshape(-1, 32, 32, 3)
                batch2_label = dataset[b"labels"]
            with open(path + "data_batch_3", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                batch3_data = dataset[b"data"].reshape(-1, 32, 32, 3)
                batch3_label = dataset[b"labels"]
            with open(path + "data_batch_4", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                batch4_data = dataset[b"data"].reshape(-1, 32, 32, 3)
                batch4_label = dataset[b"labels"]
            with open(path + "data_batch_5", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                batch5_data = dataset[b"data"].reshape(-1, 32, 32, 3)
                batch5_label = dataset[b"labels"]
            self._imgs = np.concatenate(
                [batch1_data, batch2_data, batch3_data, batch4_data, batch5_data]
            )
            self._labels = np.concatenate(
                [batch1_label, batch2_label, batch3_label, batch4_label, batch5_label]
            ).reshape(-1)

        if mode == "test":
            with open(path + "test_batch", "rb") as f:
                dataset = pickle.load(f, encoding="bytes")
                self._imgs = dataset[b"data"].reshape(-1,  32, 32,3)
                self._labels = np.array(dataset[b"labels"]).reshape(-1)

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        img = self._imgs[idx]
        label = torch.tensor(self._labels[idx]).long()

        if self._transform:
            img = self._transform(img)

        return img, label


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = TrainDatasets(transform=transform)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
