"""Gaussian actor and state-value critic for the privileged Tennis teacher."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, TanhTransform


class TeacherActorCritic(nn.Module):
    def __init__(
            self,
            state_size,
            action_size,
            hidden_size=128,
            initial_log_std=-0.5,
            normalize_observations=False,
            observation_clip=5.0,
    ):
        super().__init__()
        self.normalize_observations = normalize_observations
        self.observation_clip = observation_clip
        if normalize_observations:
            # Saved with the weights and copied to rollout workers; identity until the first update.
            self.register_buffer("observation_mean", torch.zeros(state_size))
            self.register_buffer("observation_var", torch.ones(state_size))

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1),
        )

        self.log_std = nn.Parameter(
             torch.full(
                  (action_size,),
                  initial_log_std,
                  dtype=torch.float32,
             )
        )

    def normalize(self, states):
        """Standardize raw observations with the stored running statistics."""
        if not self.normalize_observations:
            return states
        scaled = (states - self.observation_mean) / torch.sqrt(self.observation_var + 1e-8)
        return scaled.clamp(-self.observation_clip, self.observation_clip)

    def set_observation_statistics(self, mean, var):
        with torch.no_grad():
            self.observation_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
            self.observation_var.copy_(torch.as_tensor(var, dtype=torch.float32))

    def distribution(self, states):
        means = self.actor(self.normalize(states))
        std = self.log_std.exp()
        return Normal(means, std)

    def action_log_prob(self, distribution, latent_actions):
         transform = TanhTransform()
         actions = transform(latent_actions)
         gaussian_log_probs = distribution.log_prob(latent_actions)

         log_correction = transform.log_abs_det_jacobian(
              latent_actions,
              actions,
         )

         return (gaussian_log_probs - log_correction).sum(dim=-1)

    def mean_action(self, states):
        """Deterministic action: tanh of the Gaussian mean, from normalized inputs."""
        return torch.tanh(self.actor(self.normalize(states)))

    def forward(self, states):
        states = self.normalize(states)
        means = self.actor(states)
        values = self.critic(states).squeeze(-1)
        return means, values

    def sample(self, states):
        distribution = self.distribution(states)

        latent_actions = distribution.sample()
        actions = torch.tanh(latent_actions)

        log_probs = self.action_log_prob(
            distribution,
            latent_actions,
        )

        values = self.critic(self.normalize(states)).squeeze(-1)

        return latent_actions, actions, log_probs, values

    def evaluate(self, states, latent_actions):
        distribution = self.distribution(states)

        log_probs = self.action_log_prob(
            distribution,
            latent_actions,
        )

        values = self.critic(self.normalize(states)).squeeze(-1)

        return log_probs, values



class RunningObservationStatistics:
    """Running mean and variance of the 24 local values, shared by both input halves.

    The teacher input is [own 24, other 24]; both halves describe the same quantities,
    so one set of statistics is estimated from all 24-value blocks and used for both.
    """
    def __init__(self, local_size=24, halves=2):
        self.local_size = local_size
        self.halves = halves
        self.mean = np.zeros(local_size, dtype=np.float64)
        self.var = np.ones(local_size, dtype=np.float64)
        self.count = 0.0

    def update(self, states):
        batch = np.asarray(states, dtype=np.float64).reshape(-1, self.local_size)
        count = batch.shape[0]
        if count == 0:
            return
        batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)
        if self.count == 0:
            self.mean, self.var, self.count = batch_mean, batch_var, float(count)
            return
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * count / total
        m2 = self.var * self.count + batch_var * count + delta ** 2 * self.count * count / total
        self.var, self.count = m2 / total, total

    def model_statistics(self):
        return np.tile(self.mean, self.halves), np.tile(self.var, self.halves)
