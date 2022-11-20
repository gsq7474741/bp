import torchvision


class Mnist(torchvision.datasets.FashionMNIST):
    def __init__(self, root: str, size: int = 256, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.size = size

    def __len__(self):
        return self.size
