import torch
import numpy as np

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.privileged_observations = None
            self.obs_histories = None
            # self.critic_observations=None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_bins = None
            self.base_vel = None
            self.cv = None
            self.estimation_height = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        actions_shape,
        height_estimation_shape,
        device="cpu",
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.rewards = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.values = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.returns = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.advantages = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
        )
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For VAE
        self.base_vel = torch.zeros(num_transitions_per_env, num_envs, 3, device=self.device)
        self.estimation_height = torch.zeros(num_transitions_per_env, num_envs, height_estimation_shape, device=self.device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0
        self.cv = []

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.observation_histories[self.step].copy_(transition.obs_histories)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.base_vel[self.step].copy_(transition.base_vel)
        self.env_bins[self.step] = 1
        self.cv=transition.cv
        self.estimation_height *=0
        self.step += 1
        

    # def _save_hidden_states(self, hidden_states):
    #     if hidden_states is None or hidden_states == (
    #         None,
    #         None,
    #     ):
    #         return
    #     # make a tuple out of GRU hidden state sto match the LSTM format
    #     hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
    #     hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

    #     # initialize if needed
    #     if self.saved_hidden_states_a is None:
    #         self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
    #         self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
    #     # copy the states
    #     for i in range(len(hid_a)):
    #         self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
    #         self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        计算每个时间步的回报(returns)和优势估计(advantages)
        last_values: 最后一个时间步的值函数估计。
        gamma: 折扣因子, 用于计算未来奖励的当前价值。
        lam: GAE中的平滑参数, 用于权衡方差和偏差。
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """
        计算轨迹的从开始到结束的步数和所有奖励的平均值。
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成器，用于在训练过程中生成小批量数据
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        privileged_obs = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        old_env_bins = self.env_bins.flatten(0, 1)
        base_vel = self.base_vel.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        cv = self.cv
        estimation_height = self.estimation_height.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                env_bins_batch = old_env_bins[batch_idx]
                base_vel_batch = base_vel[batch_idx]
                dones_batch = dones[batch_idx]
                estimation_height_batch = estimation_height[batch_idx]
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, env_bins_batch,base_vel_batch,dones_batch,next_obs_batch, cv, estimation_height_batch