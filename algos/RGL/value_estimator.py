import torch.nn as nn
from algos.RGL.helpers import mlp


class ValueEstimator(nn.Module):
    def __init__(self, graph_model, X_dim=32, value_network_dims=[32, 100, 100, 1]):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(X_dim, value_network_dims)

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(state)[:, 0, :]
        value = self.value_network(state_embedding)
        return value
