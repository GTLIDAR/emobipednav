import torch
import torch.nn as nn
from torchviz import make_dot

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# image size is 100*100
OUT_DIM_100 = {2: 47, 4: 43, 6: 39}
# image size is 84*84
OUT_DIM_84 = {2: 39, 4: 35, 6: 31}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[1] == 100:
            out_dim = OUT_DIM_100[num_layers]
        else:
            out_dim = OUT_DIM_84[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):
        obs = obs / 255.0
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.reshape(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)

        h_norm = self.ln(h_fc)

        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass



_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
    
def main():
    encoder_type = 'pixel'
    encoder_feature_dim = 50
    num_layers = 4
    num_filters = 32
    obs_shape = (9, 100, 100)
    encoder = make_encoder(
        encoder_type, obs_shape, encoder_feature_dim, num_layers,
        num_filters
    )
    x = torch.randn(1, *obs_shape)  # Batch size = 1, Input size = obs_shape
    # output = encoder(x)
    # dot = make_dot(output, params=dict(encoder.named_parameters()))
    # dot.render("model_architecture", format="png")
    
    torch.onnx.export(
        encoder,                       # Model
        torch.randn(1, *obs_shape),          # Dummy input
        "encoder.onnx",                # Output file
        input_names=["input"],       # Input names
        output_names=["output"],     # Output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

if __name__ == "__main__":
    main()
