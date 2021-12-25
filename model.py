import torch
import torchvision.models as tvm
import torch.nn as nn
import torch
from torchsummary import summary


def get_model(name):
    if name is 'resnet':
        return Resnet50()
    else:
        raise SystemExit('Please specify a model name')


def torch_device():
    if torch.cuda.device_count() > 0:
        return 'cuda:0'
    else:
        return 'cpu'


class Resnet50:
    def __init__(self):
        self.model = tvm.resnet50(pretrained=True)
        self.features = nn.Sequential(
            *list(self.model.children)[::-1]).to(torch_device())

    def forward_pass(self, batch):
        return self.features.forward(batch).view(batch.size(0), -1)
