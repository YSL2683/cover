import torch
import torch.nn as nn


# Load the DINOv2 model only once
DINO = None


def tie_weights(src, trg):
    assert type(src) is type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=2,
        num_filters=32,
        output_logits=False,
        conv_layer_norm=False,
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.conv_layer_norm = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, (3, 3), stride=(2, 2))]
        )
        for i in range(num_layers - 1):
            self.convs.append(
                nn.Conv2d(num_filters, num_filters, (3, 3), stride=(2, 2))
            )

        x = torch.randn([1] + list(obs_shape))
        self.outputs = dict()

        out_dim = self.forward_conv(x, flatten=False).shape[-1]

        self.conv_layer_norm = conv_layer_norm
        if self.conv_layer_norm:
            print("Using LayerNorm!")
        if self.conv_layer_norm:
            self.conv_ln = nn.ModuleList(
                [nn.LayerNorm(self.outputs["conv1"].shape[1:])]
            )
            for i in range(1, self.num_layers):
                self.conv_ln.append(
                    nn.LayerNorm(self.outputs["conv%s" % (i + 1)].shape[1:])
                )

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits

    def forward_conv(self, obs, flatten=True):
        if torch.max(obs) > 1.0:
            obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = self.convs[0](obs)
        if self.conv_layer_norm:
            conv = self.conv_ln[0](conv)
        conv = torch.relu(conv)
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
            if self.conv_layer_norm:
                conv = self.conv_ln[i](conv)
            conv = torch.relu(conv)
            self.outputs["conv%s" % (i + 1)] = conv

        if flatten:
            conv = torch.flatten(conv, start_dim=1)
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param("train_encoder/conv%s" % (i + 1), self.convs[i], step)


class MocoEncoder(PixelEncoder):
    def __init__(self, num_classes):
        super().__init__(
            obs_shape=(6, 112, 112),
            feature_dim=256,
            num_layers=3,
            num_filters=32,
            output_logits=False,
            conv_layer_norm=False,
        )


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class DINOEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=2,
        num_filters=32,
        output_logits=False,
        conv_layer_norm=False,
    ):
        super().__init__()

        print("DINO ENCODER INIT")

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.fc = nn.Linear(768, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        self.dino = None
        self.outputs = dict()

        self.output_logits = output_logits

    def dino_embed(self, obs):
        if self.dino is None:
            global DINO
            if DINO is not None:
                self.dino = DINO
            else:
                self.dino = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                ).to(obs.device)
                DINO = self.dino

        with torch.no_grad():
            image1, image2 = torch.split(obs, [3, 3], dim=1)
            dino_emb1 = self.dino(image1)
            dino_emb2 = self.dino(image2)
        return torch.cat([dino_emb1, dino_emb2], dim=1)

    def forward(self, obs, detach=False):
        h = self.dino_embed(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)


_AVAILABLE_ENCODERS = {
    "pixel": PixelEncoder,
    "identity": IdentityEncoder,
    "dino": DINOEncoder,
}


def make_encoder(
    encoder_type,
    obs_shape,
    feature_dim,
    num_layers,
    num_filters,
    output_logits=False,
    conv_layer_norm=False,
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape,
        feature_dim,
        num_layers,
        num_filters,
        output_logits,
        conv_layer_norm=conv_layer_norm,
    )
