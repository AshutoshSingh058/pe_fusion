"""Fusion dataset module.

Provides :class:`FusionDataset`, a PyTorch :class:`~torch.utils.data.Dataset`
that loads pre-processed features for *multiple* modalities simultaneously
and pairs them with binary PE labels.  Each sample is a list of feature
arrays — one array per modality — which is the expected input format for the
:class:`~models.joint_fusion.JointModel`.

A convenience factory function :func:`get_fusion_dataloader` wraps the
dataset in a :class:`~torch.utils.data.DataLoader`.
"""

import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from constants import *


class FusionDataset(Dataset):
    """Dataset for multi-modality (fusion) experiments.

    Loads feature pickle files for every modality listed in *data_type* and
    organises them so that each accession number maps to a list of feature
    arrays (one per modality).  Only accessions present in **all** modalities
    are retained.
    """

    def __init__(self, data_type:list, label_path:str, split:str):
        """Initialise the fusion dataset.

        Args:
            data_type (list[str]): Ordered list of modality names to load
                (e.g. ``['Demographics', 'Vision']``).  Each name must be a
                key in ``constants.PARSED_EMR_DICT``.
            label_path (str): Path to the accession-to-label pickle mapping.
            split (str): Dataset split.  One of ``'train'``, ``'val'``,
                ``'test'``.
        """

        # get all data paths 
        all_features = []
        for data_type in data_type:
            data_path = PARSED_EMR_DICT[data_type] / f"{data_type}_{split}.pkl"
            features = pickle.load(open(data_path, "rb"))
            all_features.append(features)

        # organize input features from different modality
        self.data = defaultdict(list)
        for acc in all_features[0].keys():    # loop over accesssions
            for features in all_features:
                self.data[acc].append(np.array(features[acc], dtype="float32"))

        self.labels = pickle.load(open(label_path, "rb"))
        self.keys = [k for k in self.labels.keys() if k in self.data]
        self.feature_size = [len(feat) for feat in self.data[self.keys[0]]]
        self.split = split

    def __len__(self):
        """Return the number of samples in this split."""

        return len(self.keys)

    def __getitem__(self, idx):
        """Return the sample at position *idx*.

        Returns:
            tuple: ``(x, y)`` for train/val splits, where *x* is a list of
            ``float32`` arrays (one per modality) and *y* is a ``float32``
            array of shape ``(1,)`` with the binary label.  During the test
            split the accession string is appended: ``(x, y, accession)``.
        """

        accession = self.keys[idx]
        x = self.data[accession]
        y = self.labels[accession]

        #x = np.array(x, dtype="float32")
        y = np.array([y], dtype="float32")

        if self.split == "test":
            return x, y, accession

        return x, y


def get_fusion_dataloader(
        dataset_args:dict, 
        dataloader_args:dict
    ):
    """Construct a :class:`~torch.utils.data.DataLoader` for multi-modality data.

    Args:
        dataset_args (dict): Keyword arguments forwarded to
            :class:`FusionDataset` (``data_type``, ``label_path``, ``split``).
        dataloader_args (dict): Keyword arguments forwarded to
            :class:`~torch.utils.data.DataLoader`.

    Returns:
        torch.utils.data.DataLoader: Ready-to-use data loader.
    """
    dataset = FusionDataset(**dataset_args)
    dataloader = DataLoader(dataset, **dataloader_args)

    return dataloader
