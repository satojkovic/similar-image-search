import fire
from annoy import AnnoyIndex
import numpy as np
from skimage import io
from transform import get_transforms


def construct_ann_index(features, metric, ntrees):
    feature_dims = features[0].shape[0]
    ann = AnnoyIndex(feature_dims, metric=metric)
    for index, feature in enumerate(features):
        ann.add_item(index, feature)
    ann.build(ntrees)
    return ann


def get_single_image(query_image):
    image = io.imread(query_image)
    transform = get_transforms()
    image = transform(image)
    return image


def run(query_image, features_npy, metrics='angular', ntrees=50):
    features = np.load(features_npy)
    ann = construct_ann_index(features, metrics, ntrees)
    query_image = get_single_image(query_image)


if __name__ == '__main__':
    fire.Fire(run)
