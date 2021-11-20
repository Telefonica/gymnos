#
#
#   Dataset
#
#

from PIL import Image
from torch.utils.data import Dataset


class SAGANDataset(Dataset):

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])

        if len(img.split()) != 3:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)