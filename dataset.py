from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os


class MultiResolutionDataset_lmdb(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = r'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes) #-> get I/O
        img = Image.open(buffer) #-> to Image.object
        img = self.transform(img)

        return img


class MultiResolutionDataset_imgs(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.path = path
        files = os.listdir(path)
        self.files = [os.path.join(path, f) for f in files]

        self.length = len(self.files)

        self.resolution = resolution
        self.rescale = transforms.Compose([transforms.Resize(resolution), transforms.RandomCrop(resolution)])
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pth = self.files[index % self.length]

        img = Image.open(pth).convert('RGB') #-> to Image.object
        img = self.rescale(img)
        img = self.transform(img)

        return img