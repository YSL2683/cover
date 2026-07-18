import torch
import torch.linalg
import torch.nn.functional
import numpy as np
from torch import nn
from collections import OrderedDict
from utils import create_mlp
from encoder import tie_weights


class E2CDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_shape,
        num_layers=3,
        num_filters=64,
        n_hidden_layers=2,
        hidden_size=128,
    ):
        super().__init__()

        assert len(out_shape) == 3, "Please specify output image size, channel first."
        assert out_shape[1] % (2**num_layers) == 0, "Only supports 2x up-scales."
        self.out_shape = out_shape
        self.num_layers = num_layers

        ff_layers = OrderedDict()
        previous_feature_size = input_dim
        for i in range(n_hidden_layers):
            ff_layers[f"linear_{i + 1}"] = nn.Linear(
                in_features=previous_feature_size, out_features=hidden_size
            )
            ff_layers[f"relu_{i + 1}"] = nn.ReLU()
            previous_feature_size = hidden_size

        side_length = self.out_shape[1] // (2**self.num_layers)
        self.smallest_image_size = (num_filters, side_length, side_length)
        flattened_size = int(np.prod(self.smallest_image_size))
        ff_layers[f"linear_{n_hidden_layers + 1}"] = nn.Linear(
            in_features=previous_feature_size, out_features=flattened_size
        )
        ff_layers[f"relu_{n_hidden_layers + 1}"] = nn.ReLU()
        self.ff_layers = nn.Sequential(ff_layers)

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
            )
        self.conv_layers.append(
            nn.Conv2d(num_filters, out_shape[0], 3, stride=1, padding=1)
        )

    def forward(self, z):
        h = self.ff_layers(z)
        h = h.reshape((h.shape[0], *self.smallest_image_size))

        for i in range(self.num_layers - 1):
            h = nn.functional.interpolate(h, scale_factor=2)
            h = self.conv_layers[i](h)
            h = nn.functional.relu(h)
        output = nn.functional.interpolate(h, scale_factor=2)
        output = self.conv_layers[-1](output)
        return output


class E2CEncoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=3,
        num_filters=32,
        n_hidden_layers=2,
        hidden_size=32,
        min_log_std=-10,
        max_log_std=2,
    ):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))

        x = torch.rand([1] + list(obs_shape))
        conv_shapes = self.forward_conv(x, init=True)

        self.conv_ln = nn.ModuleList([nn.LayerNorm(s[1:]) for s in conv_shapes[:-1]])

        ff_layers = OrderedDict()
        previous_feature_size = conv_shapes[-1][1]
        for i in range(n_hidden_layers):
            ff_layers[f"linear_{i + 1}"] = nn.Linear(
                in_features=previous_feature_size, out_features=hidden_size
            )
            ff_layers[f"relu_{i + 1}"] = nn.ReLU()
            previous_feature_size = hidden_size

        ff_layers[f"linear_{n_hidden_layers + 1}"] = nn.Linear(
            in_features=previous_feature_size, out_features=2 * feature_dim
        )
        self.ff_layers = nn.Sequential(ff_layers)

    def forward_conv(self, obs, init=False):
        """
        When 'init' is set to true, this function is used to probe the shapes of intermediate results.
        """
        assert (
            obs.max() <= 1 and 0 <= obs.min()
        ), f"Make sure images are in [0, 1]. Get [{obs.min()}, {obs.max()}]"
        out_shapes = []

        conv = obs
        for i in range(self.num_layers):
            conv = self.conv_layers[i](conv)
            if not init:
                conv = self.conv_ln[i](conv)
            conv = torch.relu(conv)
            out_shapes.append(conv.shape)
        conv = conv.reshape(conv.shape[0], -1)
        out_shapes.append(conv.shape)
        if init:
            return out_shapes
        else:
            return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.ff_layers(h)
        mean, log_std = out.split([self.feature_dim, self.feature_dim], dim=1)
        log_std = log_std.clip(self.min_log_std, self.max_log_std)
        return mean, log_std

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.conv_layers[i])


class LocalLinearModel(nn.Module):
    """
    Forward model linear w.r.t. state and action.
    A, B are functions of s, parameterized by neural nets.
    """

    def __init__(self, state_dim, action_dim, n_hidden_layers=2, hidden_size=512):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ff = create_mlp(
            state_dim,
            state_dim * (2 + action_dim + 1),
            n_hidden_layers=n_hidden_layers,
            hidden_size=hidden_size,
        )

    def forward(self, s, a):
        a_matrix, b_matrix, offset = self.get_params(s)
        zp = (
            torch.bmm(a_matrix, s.unsqueeze(-1)) + torch.bmm(b_matrix, a.unsqueeze(-1))
        ).squeeze(-1) + offset
        return zp

    def get_params(self, s):
        ff_result = self.ff(s)
        a_flattened, b_flattened, offset = ff_result.split(
            [self.state_dim * 2, self.state_dim * self.action_dim, self.state_dim],
            dim=1,
        )
        a_matrix = (
            torch.eye(self.state_dim, device=s.device)
            .unsqueeze(0)
            .repeat(s.shape[0], 1, 1)
        )
        a_matrix += a_flattened[:, : self.state_dim].unsqueeze(-1) @ a_flattened[
            :, self.state_dim :
        ].unsqueeze(1)
        b_matrix = b_flattened.reshape(
            (b_flattened.shape[0], self.state_dim, self.action_dim)
        )
        return a_matrix, b_matrix, offset


class E2C(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        z_dimension: int,
        global_linear=False,
        noise=0.01,
        crop_shape=None,
    ):
        super().__init__()
        if crop_shape is None:
            in_shape = obs_shape
        else:
            in_shape = crop_shape

        self.enc = self.make_encoder(in_shape, z_dimension)
        self.dec = self.make_decoder(z_dimension, obs_shape)
        self.noise = noise
        self.crop_shape = crop_shape

        self.global_linear = global_linear
        if global_linear:
            self.fm = None
            raise NotImplementedError
        else:
            self.fm = LocalLinearModel(z_dimension, action_dim)

    def make_encoder(self, obs_shape, z_dimension):
        return E2CEncoder(obs_shape, z_dimension)

    def make_decoder(self, z_dimension, obs_shape):
        return E2CDecoder(z_dimension, obs_shape, num_filters=128)

    def forward(self, obs, action, next_obs, obs_non_crop=None, next_obs_non_crop=None):
        z_mean, z_log_std = self.enc(obs)
        next_z_mean, next_z_log_std = self.enc(next_obs)

        ref_mean = torch.zeros(z_mean.shape[1], device=z_mean.device).unsqueeze(0)
        ref_cov = torch.eye(z_mean.shape[1], device=z_mean.device).unsqueeze(0)
        ref_dist = torch.distributions.MultivariateNormal(
            ref_mean, covariance_matrix=ref_cov
        )

        z_cov = torch.diag_embed(torch.exp(z_log_std * 2))
        z_dist = torch.distributions.MultivariateNormal(z_mean, covariance_matrix=z_cov)
        next_z_cov = torch.diag_embed(torch.exp(next_z_log_std * 2))
        next_z_dist = torch.distributions.MultivariateNormal(
            next_z_mean, covariance_matrix=next_z_cov
        )

        z_sample = z_dist.rsample()
        obs_dec = self.dec(z_sample)

        a_matrices, b_matrices, offset = self.fm.get_params(z_sample)
        pred_next_z_mean = a_matrices @ z_mean.unsqueeze(
            -1
        ) + b_matrices @ action.unsqueeze(-1)
        pred_next_z_mean = pred_next_z_mean.squeeze(-1) + offset
        pred_next_z_cov = (
            a_matrices @ z_cov @ a_matrices.transpose(1, 2) + ref_cov * self.noise
        )

        pred_next_z_dist = torch.distributions.MultivariateNormal(
            pred_next_z_mean, covariance_matrix=pred_next_z_cov, validate_args={}
        )

        # Use transformed z samples, as per e2c paper Sec 2.4
        pred_next_z_sample = a_matrices @ z_sample.unsqueeze(
            -1
        ) + b_matrices @ action.unsqueeze(-1)
        pred_next_z_sample = pred_next_z_sample.squeeze(-1) + offset
        pred_next_z_dec = self.dec(pred_next_z_sample)

        dkl = torch.distributions.kl_divergence(pred_next_z_dist, next_z_dist)

        # Omits the const term in log. Using MSE, can be derived from a weighted KL.
        if self.crop_shape is None:
            if len(obs.shape) == 4:
                mse_a = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
                mse_b = 0.5 * torch.mean(
                    (next_obs - pred_next_z_dec) ** 2, dim=(1, 2, 3)
                )
            else:
                mse_a = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=1)
                mse_b = 0.5 * torch.mean((next_obs - pred_next_z_dec) ** 2, dim=1)
        else:
            mse_a = 0.5 * torch.mean((obs_non_crop - obs_dec) ** 2, dim=(1, 2, 3))
            mse_b = 0.5 * torch.mean(
                (next_obs_non_crop - pred_next_z_dec) ** 2, dim=(1, 2, 3)
            )

        ref_kl = torch.distributions.kl_divergence(z_dist, ref_dist)
        # return torch.mean(dkl), torch.mean(mse_a), torch.mean(mse_b), torch.mean(ref_kl), obs_dec, pred_next_z_dec
        return (
            torch.mean(dkl),
            torch.mean(mse_a) + torch.mean(mse_b),
            torch.mean(ref_kl),
            pred_next_z_dec,
        )

    def predict_next_obs(self, obs, action):
        z_mean, z_log_std = self.enc(obs)
        z_cov = torch.diag_embed(torch.exp(z_log_std * 2))
        z_dist = torch.distributions.MultivariateNormal(z_mean, covariance_matrix=z_cov)
        z_sample = z_dist.rsample()
        a_matrices, b_matrices, offset = self.fm.get_params(z_sample)
        pred_next_z_sample = a_matrices @ z_sample.unsqueeze(
            -1
        ) + b_matrices @ action.unsqueeze(-1)
        pred_next_z_sample = pred_next_z_sample.squeeze(-1) + offset
        pred_next_z_dec = self.dec(pred_next_z_sample)
        return pred_next_z_dec


class MLPEncoder(nn.Module):
    def __init__(self, in_size, z_dimension, n_hidden_layers=3, hidden_size=512):
        super().__init__()
        self.in_size = in_size
        self.z_dimension = z_dimension
        self.ff = create_mlp(
            in_size,
            z_dimension * 2,
            n_hidden_layers=n_hidden_layers,
            hidden_size=hidden_size,
        )

    def forward(self, obs):
        out = self.ff(obs)
        mean, log_std = out.split([self.z_dimension, self.z_dimension], dim=1)
        log_std = log_std.clip(-10, 2)
        return mean, log_std


class MLPE2C(E2C):
    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        z_dimension: int,
        global_linear=False,
        noise=0.01,
        crop_shape=None,
    ):
        super().__init__(
            obs_shape, action_dim, z_dimension, global_linear, noise, crop_shape
        )

    def make_encoder(self, obs_shape, z_dimension):
        return MLPEncoder(obs_shape[0], z_dimension, hidden_size=1024)

    def make_decoder(self, z_dimension, obs_shape):
        return create_mlp(z_dimension, obs_shape[0], hidden_size=1024)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CNN E2C
    e2c = E2C((3, 256, 256), 3, 16).to(device)
    rand_obs = torch.rand((16, 3, 256, 256)).to(device)
    rand_action = torch.rand((16, 3)).to(device)
    rand_next_obs = torch.rand((16, 3, 256, 256)).to(device)
    e2c(rand_obs, rand_action, rand_next_obs)

    # MLP E2C
    e2c = MLPE2C((384,), 3, 16).to(device)
    rand_obs = torch.rand((16, 384)).to(device)
    rand_action = torch.rand((16, 3)).to(device)
    rand_next_obs = torch.rand((16, 384)).to(device)
    e2c(rand_obs, rand_action, rand_next_obs)
