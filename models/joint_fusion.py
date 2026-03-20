"""Joint (early) fusion model module.

This module implements a joint fusion architecture where each input modality
has its own dedicated FCNN feature extractor.  The extracted representations
are concatenated and fed into a shared classifier head, also built from an
FCNN.  This design allows the model to learn modality-specific representations
before combining them for the final classification.

Typical usage::

    model = JointModel(
        feature_size=[emr_dim, vision_dim],
        num_neurons=512,
        dropout_prob=0.2,
        init_method='kaiming',
        num_hidden=2,
        activation='ReLU',
    )
    logit = model([emr_tensor, vision_tensor])
"""

import torch
import torch.nn as nn

from .fcnn import FCNN

class JointModel(nn.Module):
    """Joint fusion model that combines multiple modalities via feature concatenation.

    Each modality is first encoded by an individual :class:`FCNN` (without a
    classification head).  The resulting embeddings are concatenated along the
    feature dimension and passed to a shared :class:`FCNN` classifier.

    Args:
        feature_size (list[int]): A list of input dimensionalities, one entry
            per modality (e.g. ``[emr_dim, vision_dim]``).
        num_neurons (int): Number of neurons in every FCNN layer.
        dropout_prob (float): Dropout probability used in all FCNNs.
        init_method (str): Weight initialisation strategy (see
            :class:`FCNN`).
        num_hidden (int): Number of hidden layers in each FCNN.
        activation (str): Activation function name (see :class:`FCNN`).
    """

    def __init__(
            self, 
            feature_size:list, 
            num_neurons:int, 
            dropout_prob:float, 
            init_method:str, 
            num_hidden:int, 
            activation:str
        ):
        """Initialise the JointModel.

        Builds one feature-extraction :class:`FCNN` per modality (with
        ``classify=False``) and a single shared classifier :class:`FCNN`.

        Args:
            feature_size (list[int]): Input dimensionality for each modality.
            num_neurons (int): Neurons per layer in every FCNN.
            dropout_prob (float): Dropout probability.
            init_method (str): Weight initialisation strategy.
            num_hidden (int): Hidden layers in every FCNN.
            activation (str): Activation function name.
        """

        super(JointModel, self).__init__()

        # input args
        self.feature_size = feature_size
        self.num_neurons = num_neurons
        self.dropout_prob = dropout_prob
        self.init_method = init_method
        self.num_hidden = num_hidden
        self.activation = activation

        # feature extractor
        self.feature_fcnns = []

        # fcnn for each input modality 
        for size in feature_size:
            feature_fcnn = FCNN(
                feature_size=size, 
                num_neurons=self.num_neurons,
                num_hidden=self.num_hidden,
                init_method=self.init_method,
                activation=self.activation,
                dropout_prob=self.dropout_prob,
                classify=False
            )
            self.feature_fcnns.append(feature_fcnn)
        self.feature_fcnns = nn.ModuleList(self.feature_fcnns)

        # classifier head
        classifier_input_dim = self.num_neurons*len(feature_size)
        self.classifier_layers = FCNN(
                feature_size=classifier_input_dim, 
                num_neurons=self.num_neurons,
                num_hidden=self.num_hidden,
                init_method=self.init_method,
                activation=self.activation,
                dropout_prob=self.dropout_prob
        )

    def forward(self, x):
        """Run a forward pass through all per-modality encoders and the classifier.

        Args:
            x (list[torch.Tensor]): A list of tensors, one per modality.
                Each tensor has shape ``(batch, modality_feature_size)``.

        Returns:
            torch.Tensor: Logit tensor of shape ``(batch, 1)``.
        """

        # get features from each modality's FCNN
        features = []
        for feature, feature_fcnn in zip(x, self.feature_fcnns):
            feature = feature_fcnn(feature)
            features.append(feature)
        joint_features = torch.cat(features, 1)

        # classifier layers
        pred = self.classifier_layers(joint_features)
        return pred