from torch.utils.data import Dataset
from skimage import io
import glob
import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*', '*.jpg'))
        assert len(self.image_paths) != 0, "No images found in {}".format(
            os.path.join(root_dir, '*', '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # Return image in RGB format
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image
