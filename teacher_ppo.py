"""Shared teacher PPO with GAE and normalized advantages."""
import numpy as np
import torch

from teacher_model import RunningObservationStatistics, TeacherActorCritic

class TeacherRolloutBuffer:
    def __init__(self, rollout_steps, num_agents, state_size, action_size):
        self.rollout_steps = rollout_steps
        self.position = 0

        self.states = np.zeros(
            (rollout_steps, num_agents, state_size),
            dtype=np.float32,
        )

        self.latent_actions = np.zeros(
            (rollout_steps, num_agents, action_size),
            dtype=np.float32,
        )

        scalar_shape = (rollout_steps, num_agents)

        self.rewards = np.zeros(scalar_shape, dtype=np.float32)
        self.values = np.zeros(scalar_shape, dtype=np.float32)
        self.log_probs = np.zeros(scalar_shape, dtype=np.float32)
        self.next_values = np.zeros(scalar_shape, dtype=np.float32)

        self.terminated = np.zeros(scalar_shape, dtype=bool)
        self.ended = np.zeros(scalar_shape, dtype=bool)

    def add(self, states, latent_actions, rewards, values, log_probs,
            next_values, terminated, ended):
        if self.position >= self.rollout_steps:
            raise RuntimeError("The rollout buffer is full.")

        i = self.position

        self.states[i] = states
        self.latent_actions[i] = latent_actions
        self.rewards[i] = rewards
        self.values[i] = values
        self.log_probs[i] = log_probs
        self.next_values[i] = next_values
        self.terminated[i] = terminated
        self.ended[i] = ended

        self.position += 1

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute GAE per player, bootstrapping truncations but not terminals."""
        count = self.position
        if count == 0:
            raise RuntimeError("Cannot calculate returns from an empty rollout.")

        advantages = np.zeros_like(self.rewards[:count])
        next_advantage = np.zeros_like(self.rewards[0])
        for t in reversed(range(count)):
            bootstrap_value = np.where(self.terminated[t], 0.0, self.next_values[t])
            delta = self.rewards[t] + gamma * bootstrap_value - self.values[t]
            continuation = np.where(
                self.ended[t] | self.terminated[t], 0.0, next_advantage
            )
            advantages[t] = delta + gamma * gae_lambda * continuation
            next_advantage = advantages[t]

        returns = advantages + self.values[:count]
        return advantages, returns

class TeacherPPOAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 hidden_size=128,
                 learning_rate=0.0003,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 update_epochs=10,
                 minibatch_size=512,
                 value_loss_coef=0.5,
                 gae_lambda=0.95,
                 normalize_observations=False,
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.value_loss_coef = value_loss_coef
        self.gae_lambda = gae_lambda

        self.model = TeacherActorCritic(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            normalize_observations=normalize_observations,
        )
        self.observation_statistics = (
            RunningObservationStatistics() if normalize_observations else None
        )

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=learning_rate,
        )


    def act(self, states):
        state_tensor = torch.as_tensor(
            data=states,
            dtype=torch.float32,
        )

        with torch.no_grad():
            latent_actions, actions, log_probs, values = self.model.sample(
                states=state_tensor,
            )

        return (
            latent_actions.numpy(),
            actions.numpy(),
            log_probs.numpy(),
            values.numpy(),
        )

    def compute_loss(
            self,
            states,
            latent_actions,
            old_log_probs,
            advantages,
            returns,
    ):
        new_log_probs, new_values = self.model.evaluate(
            states=states,
            latent_actions=latent_actions,
        )

        ratios = torch.exp(
            input=new_log_probs - old_log_probs,
        )

        surrogate = ratios * advantages

        clipped_surrogate = torch.clamp(
            input=ratios,
            min=1.0 - self.clip_epsilon,
            max=1.0 + self.clip_epsilon,
        ) * advantages

        policy_loss = -torch.minimum(
            input=surrogate,
            other=clipped_surrogate,
        ).mean()

        value_loss = (new_values - returns).square().mean()
        total_loss = policy_loss + self.value_loss_coef * value_loss

        return total_loss, policy_loss, value_loss

    def update(self, rollout):
        advantages, returns = rollout.compute_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        count = rollout.position

        states = torch.as_tensor(
            data=rollout.states[:count],
            dtype=torch.float32,
        ).flatten(start_dim=0, end_dim=1)

        latent_actions = torch.as_tensor(
            data=rollout.latent_actions[:count],
            dtype=torch.float32,
        ).flatten(start_dim=0, end_dim=1)

        old_log_probs = torch.as_tensor(
            data=rollout.log_probs[:count],
            dtype=torch.float32,
        ).flatten()

        advantages = torch.as_tensor(
            data=advantages,
            dtype=torch.float32,
        ).flatten()

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        returns = torch.as_tensor(
            data=returns,
            dtype=torch.float32,
        ).flatten()

        sample_count = states.shape[0]

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        processed_samples = 0
        optimizer_steps = 0

        for epoch in range(self.update_epochs):
            indices = torch.randperm(
                n=sample_count,
            )

            for start in range(0, sample_count, self.minibatch_size):
                batch_indices = indices[
                    start:start + self.minibatch_size
                ]

                total_loss, policy_loss, value_loss = self.compute_loss(
                    states=states[batch_indices],
                    latent_actions=latent_actions[batch_indices],
                    old_log_probs=old_log_probs[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=returns[batch_indices],
                )

                if not torch.isfinite(input=total_loss):
                    raise FloatingPointError("PPO loss became non-finite.")

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                batch_size = len(batch_indices)

                policy_loss_sum += policy_loss.item() * batch_size
                value_loss_sum += value_loss.item() * batch_size
                processed_samples += batch_size
                optimizer_steps += 1

        if self.observation_statistics is not None:
            # Collection and this update used the same statistics, so PPO ratios start at 1.
            # The new statistics take effect from the next rollout onward.
            self.observation_statistics.update(rollout.states[:count])
            self.model.set_observation_statistics(*self.observation_statistics.model_statistics())

        return {
            "policy_loss": policy_loss_sum / processed_samples,
            "value_loss": value_loss_sum / processed_samples,
            "optimizer_steps": optimizer_steps,
        }
