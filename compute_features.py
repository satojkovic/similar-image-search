import fire
import os

from dataset import ImageDataset
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from torchvision import transforms
from transform import Rescale, RandomCrop, ToTensor
import os


def run(model_name, output_dir, dataname, data_dir, batch_size=8):
    data_path = os.path.join(data_dir, dataname)
    ds = ImageDataset(data_path, transform=transforms.Compose(
        [Rescale(256), RandomCrop(224), ToTensor()]))
    model = get_model(model_name)
    data_loader = DataLoader(ds, batch_size=batch_size)

    features_list = []
    iterator = tqdm(data_loader)
    for batch in iterator:
        output = model.forward_pass(batch.to(torch_device()))
        features_list.append(output.cpu().detach().numpy())

    features = np.vstack(features_list)
    output_path = '%s/%s-%s--%s' % (output_dir, model_name,
                                    dataname, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, features)


if __name__ == '__main__':
    fire.Fire(run)
