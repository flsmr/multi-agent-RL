"""Gaussian actor and state-value critic for the privileged Tennis teacher."""
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
    ):
        super().__init__()

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

    def distribution(self, states):
        means = self.actor(states)
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

    def forward(self, states):
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

        values = self.critic(states).squeeze(-1)

        return latent_actions, actions, log_probs, values

    def evaluate(self, states, latent_actions):
        distribution = self.distribution(states)

        log_probs = self.action_log_prob(
            distribution,
            latent_actions,
        )

        values = self.critic(states).squeeze(-1)

        return log_probs, values

