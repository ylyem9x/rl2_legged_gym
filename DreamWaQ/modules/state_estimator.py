import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Normal
from torch.nn import functional

class VAE(nn.Module):
    def __init__(
        self,
        num_obs,
        num_history,
        num_latent,
        num_est,
        activation="elu",
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[512, 256, 128],
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_his = num_history
        self.num_latent = num_latent
        self.num_est = num_est

        self.encoder = encoder(
            num_history * num_obs,
            self.num_latent * 2 + self.num_est * 2, # 4 is the front height
            activation,
            encoder_hidden_dims,
        )

        self.decoder = decoder(
            self.num_latent + self.num_est,
            num_obs,# num_obs
            activation,
            decoder_hidden_dims,
        )


    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = encoded[:,:self.num_latent]
        latent_var = encoded[:,self.num_latent:self.num_latent * 2]
        est_mu = encoded[:, self.num_latent * 2:self.num_latent * 2 + self.num_est]
        est_var = encoded[:, self.num_latent * 2 + self.num_est:self.num_latent * 2 + self.num_est * 2]
        return [latent_mu, latent_var, est_mu, est_var]

    def decode(self, z, est):
        input = torch.cat([z, est], dim=1)
        output = self.decoder(input)
        return output

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return: eps * std + mu
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs_history):
        latent_mu, latent_var, est_mu, est_var = self.encode(obs_history)
        latent_var = torch.clip(latent_var, max = 0.5)
        est_var = torch.clip(est_var, max = 0.5)
        z = self.reparameterize(latent_mu, latent_var)
        est = self.reparameterize(est_mu, est_var)
        return [z, est], [latent_mu, latent_var, est_mu, est_var]

    def loss_fn(self, obs_history, obs_next, real, kld_weight=1.0):
        [z, est], [latent_mu, latent_var, est_mu, est_var] = self.forward(obs_history)

        # Body velocity estimation loss
        est_loss = functional.mse_loss(real, est, reduction="none").mean(-1).clamp(0, 1.0)
        # MSE of obs in VAE loss
        recons_obs = self.decode(z, est)
        recons_loss = functional.mse_loss(recons_obs, obs_next, reduction="none").mean( #obs_next
            -1
        ).clamp(0, 1.0)

        # KL divergence as latent loss
        # KL in VAE = -0.5sum(1+log(σ^2)-miu^2-σ^2)
        kld_loss = -0.5 * torch.sum(
            1 + latent_var - latent_mu**2 - latent_var.exp(), dim=1
        )

        loss = recons_loss + 5*est_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recons_loss": recons_loss,
            "est_loss": est_loss,
            "kld_loss": kld_loss,
        }

    def sample(self, obs_history):
        """
        :return estimation = [z, vel]
        :dim(z) = num_latent
        :dim(vel) = 3
        """
        estimation, output = self.forward(obs_history)
        return estimation, output

    def inference(self, obs_history):
        """
        return [latent_mu, vel_mu, h_mu]
        """
        _, latent_params = self.forward(obs_history)
        latent_mu, latent_var, est_mu, est_var = latent_params
        return [latent_mu, est_mu]
    
    def inference_action(self, obs_history):
        """
        return reconstructed action
        """
        [z, est], [latent_mu, latent_var, est_mu, est_var] = self.forward(obs_history)
        recons_obs = self.decode(latent_mu, est_mu)
        action = recons_obs[:, -12:]
        return action
        


class encoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(RS_obs_history)


class decoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.decoder = nn.Sequential(*module)

    def forward(self, input):
        return self.decoder(input)
