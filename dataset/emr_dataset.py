"""EMR (Electronic Medical Record) dataset module.

Provides :class:`EMRDataset`, a PyTorch :class:`~torch.utils.data.Dataset`
that loads pre-processed EMR features from pickle files and pairs them with
binary PE (Pulmonary Embolism) labels.  A convenience factory function
:func:`get_emr_dataloader` wraps the dataset in a
:class:`~torch.utils.data.DataLoader`.

Each pickle file is expected to be a ``dict`` mapping accession numbers to
feature vectors (lists or arrays of floats).  Labels are loaded from a
separate pickle file that maps accession numbers to binary integers
(1 = PE-positive, 0 = PE-negative).
"""

import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from constants import *


class EMRDataset(Dataset):
    """Dataset for a single EMR modality (e.g. Demographics, ICD codes, Labs).

    Loads the split-specific pickle file for *data_type* and the label
    mapping from *label_path*, then exposes each accession as an
    ``(x, y)`` pair (or ``(x, y, accession)`` during the test split so that
    predictions can be traced back to individual patients).
    """

    def __init__(self, data_type:str, label_path:str, split:str):
        """Initialise the dataset.

        Args:
            data_type (str): EMR modality name (e.g. ``'Demographics'``,
                ``'ICD'``, ``'LABS'``).  Must be a key in
                ``constants.PARSED_EMR_DICT``.
            label_path (str): Path to the accession-to-label pickle mapping.
            split (str): Dataset split to load.  One of ``'train'``,
                ``'val'``, ``'test'``.
        """

        self.data_path = PARSED_EMR_DICT[data_type] / f"{data_type}_{split}.pkl"
        self.data = pickle.load(open(self.data_path, "rb"))
        self.labels = pickle.load(open(label_path, "rb"))
        self.keys = [k for k in self.labels.keys() if k in self.data]
        self.feature_size = len(self.data[self.keys[0]])
        self.split = split

    def __len__(self):
        """Return the number of samples in this split."""

        return len(self.keys)

    def __getitem__(self, idx):
        """Return the sample at position *idx*.

        Returns:
            tuple: ``(x, y)`` for training/validation splits, where *x* is a
            ``float32`` array of shape ``(feature_size,)`` and *y* is a
            ``float32`` array of shape ``(1,)`` containing the binary label.
            During the test split an accession string is appended:
            ``(x, y, accession)``.
        """

        accession = self.keys[idx]
        x = self.data[accession]
        y = self.labels[accession]

        x = np.array(x, dtype="float32")
        y = np.array([y], dtype="float32")

        if self.split == "test":
            return x, y, accession

        return x, y


def get_emr_dataloader(
        dataset_args:dict, 
        dataloader_args:dict
    ):
    """Construct a :class:`~torch.utils.data.DataLoader` for a single EMR modality.

    Args:
        dataset_args (dict): Keyword arguments forwarded to
            :class:`EMRDataset` (``data_type``, ``label_path``, ``split``).
        dataloader_args (dict): Keyword arguments forwarded to
            :class:`~torch.utils.data.DataLoader`
            (``batch_size``, ``num_workers``, ``shuffle``, …).

    Returns:
        torch.utils.data.DataLoader: Ready-to-use data loader.
    """
    dataset = EMRDataset(**dataset_args)
    dataloader = DataLoader(dataset, **dataloader_args)

    return dataloader
