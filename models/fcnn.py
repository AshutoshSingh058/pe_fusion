"""Fully Connected Neural Network (FCNN) module.

This module defines a configurable feed-forward neural network used as both
a standalone EMR classifier and as a building block inside the joint fusion
model.  The number of hidden layers, neurons per layer, activation function,
dropout probability and weight-initialisation strategy are all configurable
through constructor arguments so that the same class can be reused across
every EMR modality and fusion architecture.
"""

import torch.nn as nn


class FCNN(nn.Module):
    """Configurable fully connected feed-forward neural network.

    The network consists of:
      - One input linear layer (``feature_size`` → ``num_neurons``).
      - ``num_hidden`` hidden linear layers (``num_neurons`` → ``num_neurons``)
        each followed by an activation and a dropout layer.
      - One output linear layer (``num_neurons`` → 1) when ``classify=True``.

    When ``classify=False`` the output layer is omitted and the network acts
    as a feature-extraction backbone (used inside :class:`JointModel`).
    """

    def __init__(
            self, 
            feature_size:int, 
            num_neurons:int, 
            num_hidden:int, 
            init_method:str, 
            activation:str,
            dropout_prob:float = 0, 
            classify:bool=True
        ):
        """Initialise the FCNN.

        Args:
            feature_size (int): Dimensionality of the input feature vector.
            num_neurons (int): Number of neurons in every hidden layer.
            num_hidden (int): Number of hidden layers between the input and
                output layers.
            init_method (str): Weight initialisation strategy. Supported
                values are ``'normal'``, ``'xavier'``, and ``'kaiming'``.
                Pass ``None`` to keep PyTorch's default initialisation.
            activation (str): Non-linear activation to use after each linear
                layer.  Supported values: ``'ELU'``, ``'LeakyReLU'``,
                ``'Tanh'``, and (default) ``'ReLU'``.
            dropout_prob (float): Dropout probability applied after each
                hidden activation.  Defaults to 0 (no dropout).
            classify (bool): When ``True`` (default) a final linear layer
                that maps to a single logit is appended.  Set to ``False``
                when the network is used as a feature extractor.
        """

        super(FCNN, self).__init__()

        # input args
        self.feature_size = feature_size
        self.num_neurons = num_neurons
        self.dropout_prob = dropout_prob
        self.init_method = init_method

        # get activation
        self.get_activation(activation)

        # input layer
        self.feed_forward = nn.ModuleList()
        self.feed_forward.extend([nn.Linear(feature_size, self.num_neurons), self.activation()])

        # hidden layers
        for _ in range(num_hidden):
            self.feed_forward.extend([
                nn.Linear(self.num_neurons, self.num_neurons), \
                self.activation(), \
                nn.Dropout(self.dropout_prob)
            ])

        # output layer
        if classify:
            self.feed_forward.extend([nn.Linear(self.num_neurons, 1)])

        # init weight
        if self.init_method is not None:
            self.initialize_weights(self.init_method)

    def initialize_weights(self, init_method, gain=0.2):
        """Initialise all ``nn.Linear`` weights in the network.

        Biases are always set to zero.  Weights are initialised according to
        *init_method*:

        * ``'normal'``  – Gaussian with mean 0 and std *gain*.
        * ``'xavier'``  – Xavier (Glorot) normal initialisation scaled by *gain*.
        * ``'kaiming'`` – Kaiming (He) normal initialisation (recommended for
          ReLU activations).

        Args:
            init_method (str): One of ``'normal'``, ``'xavier'``,
                ``'kaiming'``.
            gain (float): Scaling factor used by ``'normal'`` and
                ``'xavier'`` initialisations.  Defaults to 0.2.

        Raises:
            NotImplementedError: If *init_method* is not one of the
                supported strategies.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(init_method))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_activation(self, activation):
        """Store the activation-function *class* that will be instantiated in layers.

        The chosen class is stored as ``self.activation`` and instantiated
        (without arguments) when building each layer.  Defaults to
        :class:`torch.nn.ReLU` for unrecognised values.

        Args:
            activation (str): Activation name.  Supported values:
                ``'ELU'``, ``'LeakyReLU'``, ``'Tanh'``, ``'ReLU'``.
        """
        if activation == "ELU":
            self.activation =  nn.ELU
        elif activation == "LeakyReLU":
            self.activation =  nn.LeakyReLU
        elif activation == "Tanh":
            self.activation =  nn.Tanh
        else:
            self.activation =  nn.ReLU

    def forward(self, x):
        """Run a forward pass through all layers sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, feature_size)``.

        Returns:
            torch.Tensor: Output logit tensor of shape ``(batch, 1)`` when
            ``classify=True``, or ``(batch, num_neurons)`` when
            ``classify=False``.
        """
        for layer in self.feed_forward:
            x = layer(x)
        return x
