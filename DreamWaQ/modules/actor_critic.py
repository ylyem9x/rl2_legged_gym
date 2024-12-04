import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .state_estimator import VAE


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_obs,
        num_privileged_obs,
        num_actions=12,
        num_latent=16,
        num_history=20,
        num_est=3,
        activation="elu",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 64],
        decoder_hidden_dims=[64, 128, 256],
        init_noise_std=1.0,
    ):
        super().__init__()

        self.activation = get_activation(activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.vae = VAE(
            num_obs=num_obs,
            num_history=num_history,
            num_latent=num_latent,
            num_est=num_est,
            activation=self.activation,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
        )

        self.actor = Actor(
            input_size=num_obs + num_latent + num_est,
            output_size=num_actions,
            activation=self.activation,
            hidden_dims=actor_hidden_dims,
        )
        """
        :input = obs + latent + estimated_vel
        :output = actions
        """
        
        real_vel = 3
        self.critic = Critic(
            input_size=num_obs + real_vel + num_privileged_obs,
            output_size=1,
            activation=self.activation,
            hidden_dims=critic_hidden_dims,
        )
        """
        :input = obs + real_vel + privileged_obs
        :output = reward
        """

        self.distribution = None
        
        # 在确定代码无误后，禁用参数验证可以略微提高性能
        Normal.set_default_validate_args = False

        self.vae.apply(init_orhtogonal)
        # self.actor.apply(init_orhtogonal)
        # self.critic.apply(init_orhtogonal)
        self.pre_std = torch.ones(1)

    #! rollout 的时候需要随机性, 这里是 bootstrap
    def act_student(self, obs, obs_history):
        """
        :obs_dict: obs, obs_history
        :return distribution.sample()
        :其中std会在模型的训练过程中自动调整
        """
        [z, est], [latent_mu, latent_var, est_mu, est_var] = self.vae.sample(obs_history)
        latent_and_estimated = torch.cat([z, est], dim=1)
        input = torch.cat([obs,latent_and_estimated],dim=1)
        action_mean = self.actor.forward(input)
        self.update_distribution(action_mean)
        return self.distribution.sample()

    def reset(self, dones=None):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        latent_mu, est_mu = self.vae.inference(obs_dict["obs_history"])
        latent = torch.cat([latent_mu, est_mu], dim=1)
        input = torch.cat([obs_dict["obs"], latent],dim=1)
        return self.actor.forward(input)
    
    def act_inference_vae(self, obs_dict):
        action = self.vae.inference_action(obs_dict["obs_history"])
        return action

    def evaluate(self, obs, privileged_observations, vel):
        obs = torch.cat([obs, vel], dim=-1)
        input = torch.cat([obs, privileged_observations],dim=1)
        value = self.critic.forward(input)
        return value
    
    def update_distribution(self, mean):
        self.distribution = Normal(mean, mean*0. + self.std)

class Actor(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
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
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


class Critic(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
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
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "identity":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None

def init_orhtogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)