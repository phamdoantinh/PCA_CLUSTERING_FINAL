from torch.utils.data import Dataset


class RSSIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        return X, Y