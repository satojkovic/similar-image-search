from fastai import conv_learner as learner
from torch.utils.data import Dataset

import glob
import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, tfms):
        self.image_paths = glob.glob(root_dir + '*.jpg')
        assert len(self.image_paths) != 0, "No images found in {}".format(
            root_dir)
        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.tfms = tfms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # Return image in RGB format, each pixel ranges between 0.0 and 1.0
        image = learner.open_image(image_path)
        # Apply transforms to the image
        image = self.tfms(image)
        return image
