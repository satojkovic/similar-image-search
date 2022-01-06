import fire
from annoy import AnnoyIndex
import numpy as np


def construct_ann_index(features, metric, ntrees):
    feature_dims = features[0].shape[0]
    ann = AnnoyIndex(feature_dims, metric=metric)
    for index, feature in enumerate(features):
        ann.add_item(index, feature)
    ann.build(ntrees)
    return ann


def run(features_npy, metrics='angular', ntrees=50):
    features = np.load(features_npy)
    ann = construct_ann_index(features, metrics, ntrees)


if __name__ == '__main__':
    fire.Fire(run)
