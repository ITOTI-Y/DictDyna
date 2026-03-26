"""Replay buffer implementations for Dyna-SAC."""

import numpy as np
import torch


class ReplayBuffer:
    """Standard replay buffer for RL transitions.

    Args:
        capacity: Maximum number of transitions.
        state_dim: State dimension.
        action_dim: Action dimension.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        next_state: np.ndarray,
        done: bool | np.ndarray,
    ) -> None:
        """Add a single transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.float32(reward).item()
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = np.float32(done).item()
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions."""
        n = len(states)
        for i in range(n):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(
        self, batch_size: int, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones.
        """
        device = device or torch.device("cpu")
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_states": torch.tensor(self.next_states[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample consecutive transition sequences that don't cross episode boundaries.

        A sequence [i, i+1, ..., i+seq_len-1] is valid only if dones[i] through
        dones[i+seq_len-2] are all 0 (the last step may be terminal).

        Returns:
            Dict with keys: states, actions, next_states, dones.
            Each has shape (batch_size, seq_len, dim).
        """
        device = device or torch.device("cpu")
        if self.size < seq_len:
            raise ValueError(f"Buffer size {self.size} too small for seq_len {seq_len}")

        # Vectorized episode boundary detection via cumsum
        # For start i, sequence covers [i, i+seq_len-1].
        # Valid iff done_flags[i..i+seq_len-2] are all 0.
        done_flags = (self.dones[: self.size].flatten() > 0.5).astype(np.float32)
        if seq_len > 1:
            window = seq_len - 1
            cs = np.cumsum(np.concatenate([[0.0], done_flags]))
            # window_sum[i] = sum(done_flags[i : i+window]) for i in [0, size-window]
            n_windows = self.size - window + 1
            window_sum = cs[window : window + n_windows] - cs[:n_windows]
            valid_mask = np.zeros(self.size, dtype=bool)
            valid_mask[:n_windows] = window_sum < 0.5
        else:
            valid_mask = np.ones(self.size, dtype=bool)

        # Can't start a sequence in the last seq_len-1 positions
        valid_mask[max(0, self.size - seq_len + 1) :] = False

        # If buffer has wrapped, exclude sequences crossing the write pointer
        if self.pos < self.size and self.pos > 0:
            wrap_start = max(0, self.pos - seq_len + 1)
            valid_mask[wrap_start : self.pos] = False

        valid_starts = np.where(valid_mask)[0]
        if len(valid_starts) == 0:
            raise ValueError(
                f"No valid sequences of length {seq_len} "
                f"(buffer_size={self.size}, check episode boundaries)"
            )

        chosen = np.random.choice(
            len(valid_starts), batch_size, replace=len(valid_starts) < batch_size
        )
        starts = valid_starts[chosen]

        offsets = np.arange(seq_len)
        indices = starts[:, None] + offsets[None, :]

        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "next_states": torch.tensor(self.next_states[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }

    def __len__(self) -> int:
        return self.size


class MixedReplayBuffer:
    """Mixed buffer combining real and simulated transitions.

    Args:
        real_capacity: Capacity for real transitions.
        model_capacity: Capacity for model-generated transitions.
        state_dim: State dimension.
        action_dim: Action dimension.
    """

    def __init__(
        self,
        real_capacity: int,
        model_capacity: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.real_buffer = ReplayBuffer(real_capacity, state_dim, action_dim)
        self.model_buffer = ReplayBuffer(model_capacity, state_dim, action_dim)

    def add_real(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a real environment transition."""
        self.real_buffer.add(state, action, reward, next_state, done)

    def add_model(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a model-generated transition."""
        self.model_buffer.add(state, action, reward, next_state, done)

    def clear_model_buffer(self) -> None:
        """Clear all model-generated transitions (MBPO-style refresh)."""
        self.model_buffer.pos = 0
        self.model_buffer.size = 0

    def add_model_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of model-generated transitions."""
        self.model_buffer.add_batch(states, actions, rewards, next_states, dones)

    def sample(
        self,
        batch_size: int,
        model_ratio: float = 0.5,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a mixed batch from real and model buffers.

        Args:
            batch_size: Total batch size.
            model_ratio: Fraction of samples from model buffer.
            device: Torch device.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones.
        """
        device = device or torch.device("cpu")
        n_model = int(batch_size * model_ratio) if len(self.model_buffer) > 0 else 0
        n_real = batch_size - n_model

        # Ensure we have enough data
        n_real = min(n_real, len(self.real_buffer))
        n_model = min(n_model, len(self.model_buffer))

        if n_real == 0 and n_model == 0:
            raise ValueError("Both buffers are empty")

        batches = []
        if n_real > 0:
            batches.append(self.real_buffer.sample(n_real, device))
        if n_model > 0:
            batches.append(self.model_buffer.sample(n_model, device))

        if len(batches) == 1:
            return batches[0]

        return {key: torch.cat([b[key] for b in batches], dim=0) for key in batches[0]}

    @property
    def real_size(self) -> int:
        return len(self.real_buffer)

    @property
    def model_size(self) -> int:
        return len(self.model_buffer)


class TaggedReplayBuffer:
    """Replay buffer with building_id tags for multi-building training.

    Supports sampling all data (for shared SAC) or filtered by tag
    (for per-building world model training).

    Args:
        capacity: Maximum number of transitions.
        state_dim: State dimension.
        action_dim: Action dimension.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.tags = np.full(capacity, -1, dtype=np.int32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        next_state: np.ndarray,
        done: bool | np.ndarray,
        tag: int = 0,
    ) -> None:
        """Add a single transition with building tag."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.float32(reward).item()
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = np.float32(done).item()
        self.tags[self.pos] = tag
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample from ALL data regardless of tag (for shared SAC)."""
        device = device or torch.device("cpu")
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_states": torch.tensor(self.next_states[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }

    def sample_tagged(
        self, batch_size: int, tag: int, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample only transitions with a specific building tag."""
        device = device or torch.device("cpu")
        valid = np.where(self.tags[: self.size] == tag)[0]
        if len(valid) == 0:
            raise ValueError(f"No data for tag {tag}")
        indices = valid[np.random.randint(0, len(valid), size=batch_size)]
        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_states": torch.tensor(self.next_states[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }

    def tag_count(self, tag: int) -> int:
        """Count transitions with a specific tag."""
        return int((self.tags[: self.size] == tag).sum())

    def __len__(self) -> int:
        return self.size
