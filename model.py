import timm
from torchsummary import summary


def setup_model(model_name):
    assert model_name == 'resnet50' or model_name == 'inception_v4', 'Model name must be either resnet50 or inception_v4.'
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model


if __name__ == '__main__':
    model = setup_model('resnet50')
    summary(model, input_size=(3, 224, 224))
