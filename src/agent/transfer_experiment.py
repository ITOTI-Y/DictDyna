"""Phase 4: Few-shot transfer experiment.

Train on source buildings (hot + mixed), then transfer to target (cool)
with limited data (1/3/7 days). Compare:
1. Shared D + adapter fine-tune (freeze D + trunk, train adapter only)
2. Train from scratch (independent, no transfer)
"""

import contextlib
import json
from pathlib import Path

import gymnasium
import numpy as np
import torch
from loguru import logger

from src.agent._share import normalize_obs
from src.agent.dyna_sac import DynaSAC
from src.schemas import TrainSchema
from src.utils import get_device, seed_everything, sinergym_workdir

with contextlib.suppress(ImportError):
    import sinergym  # noqa: F401


class FewShotTransferExperiment:
    """Run few-shot transfer experiment.

    Phase 1: Train on source buildings (full data)
    Phase 2: Collect limited data from target building
    Phase 3: Transfer (adapt only) vs train from scratch

    Args:
        source_configs: Source building configs [{env_name, building_id}].
        target_config: Target building config {env_name, building_id}.
        dict_path: Pretrained dictionary path.
        config: TrainSchema.
        adaptation_days: List of days to test [1, 3, 7].
        seed: Random seed.
        save_dir: Output directory.
    """

    def __init__(
        self,
        source_configs: list[dict],
        target_config: dict,
        dict_path: str,
        config: TrainSchema,
        adaptation_days: list[int] | None = None,
        seed: int = 42,
        save_dir: str = "output/results/transfer",
        context_mode: bool = False,
    ) -> None:
        self.source_configs = source_configs
        self.target_config = target_config
        self.adaptation_days = adaptation_days or [1, 3, 7]
        self.seed = seed
        self.context_mode = context_mode
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = get_device(config.device)

        seed_everything(seed)

        # Load dictionary and obs stats
        self._dict_path = dict_path
        dict_data = torch.load(dict_path, weights_only=False)
        self.dictionary = dict_data["dictionary"]
        self._obs_mean = dict_data["obs_mean"].numpy()
        self._obs_std = dict_data["obs_std"].numpy()
        self._obs_mean_t = dict_data.get("obs_mean")
        self._obs_std_t = dict_data.get("obs_std")

        # Probe env dims
        with sinergym_workdir():
            probe = gymnasium.make(target_config["env_name"])
        self.state_dim = probe.observation_space.shape[0]  # ty: ignore[not-subscriptable]
        self.action_dim = probe.action_space.shape[0]  # ty: ignore[not-subscriptable]
        action_low = probe.action_space.low  # ty: ignore[unresolved-attribute]
        action_high = probe.action_space.high  # ty: ignore[unresolved-attribute]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        probe.close()

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        return normalize_obs(raw, self._obs_mean, self._obs_std)

    def run(self) -> dict:
        """Run the full transfer experiment."""
        results: dict = {}

        # Phase 1: Train on source buildings (1 episode each)
        logger.info("=== Phase 1: Training on source buildings ===")
        source_dyna = self._train_source()

        # Phase 2: Collect target building data (full episode for eval)
        logger.info("=== Phase 2: Collecting target building data ===")
        target_data = self._collect_target_data()

        # Phase 3: For each adaptation budget
        for days in self.adaptation_days:
            steps = days * 96  # 96 steps per day at 15-min interval
            logger.info(f"\n=== Phase 3: {days}-day adaptation ({steps} steps) ===")

            # 3a: Transfer
            if self.context_mode:
                transfer_reward = self._run_context_transfer(
                    source_dyna, target_data, steps
                )
            else:
                transfer_reward = self._run_transfer(source_dyna, target_data, steps)
            results[f"transfer_{days}d"] = transfer_reward

            # 3b: Train from scratch (no transfer)
            scratch_reward = self._run_from_scratch(target_data, steps)
            results[f"scratch_{days}d"] = scratch_reward

            adv = (transfer_reward - scratch_reward) / abs(scratch_reward) * 100
            logger.info(
                f"  {days}d: Transfer={transfer_reward:.1f}, "
                f"Scratch={scratch_reward:.1f}, Advantage={adv:+.1f}%"
            )

        # Save results
        with open(self.save_dir / "transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {self.save_dir}")

        return results

    def _train_source(self) -> DynaSAC:
        """Train Dyna-SAC on source buildings (1 episode each)."""

        building_ids = [c["building_id"] for c in self.source_configs]

        dyna = DynaSAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            building_ids=building_ids,
            dictionary=self.dictionary,
            config=self.config,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            obs_mean=self._obs_mean_t,
            obs_std=self._obs_std_t,
        )

        global_step = 0
        for cfg in self.source_configs:
            bid = cfg["building_id"]

            logger.info(f"Training on {bid}...")

            with sinergym_workdir():
                env = gymnasium.make(cfg["env_name"])
            raw_obs, _ = env.reset(seed=self.seed)
            obs = self._normalize(raw_obs)
            done = False
            ep_reward = 0.0

            while not done:
                global_step += 1
                if global_step < 500:
                    action = env.action_space.sample()
                else:
                    action = dyna.select_action(obs)

                raw_next, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_obs = self._normalize(raw_next)

                if global_step >= 500:
                    dyna.train_step(
                        obs,
                        action,
                        float(reward),
                        next_obs,
                        done,
                        bid,
                        global_step,
                    )
                else:
                    dyna.buffer.add_real(obs, action, float(reward), next_obs, done)

                ep_reward += float(reward)
                obs = next_obs

            env.close()
            logger.info(f"  {bid}: reward={ep_reward:.1f}, steps={global_step}")
            dyna.on_episode_end()

        return dyna

    def _train_source_no_rollout(self) -> DynaSAC:
        """Train SAC on source buildings WITHOUT model rollouts (ablation D).

        Same env interaction and SAC updates as _train_source, but
        rollout_start_step is set to infinity so no model data is generated.
        Policy learns purely from real environment transitions.
        """
        building_ids = [c["building_id"] for c in self.source_configs]

        # Override rollout_start_step to disable rollouts entirely
        no_rollout_config = self.config.model_copy(
            update={
                "dyna": self.config.dyna.model_copy(
                    update={"rollout_start_step": 999_999_999}
                )
            }
        )
        dyna = DynaSAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            building_ids=building_ids,
            dictionary=self.dictionary,
            config=no_rollout_config,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            obs_mean=self._obs_mean_t,
            obs_std=self._obs_std_t,
        )

        global_step = 0
        for cfg in self.source_configs:
            bid = cfg["building_id"]
            logger.info(f"Training on {bid} (no rollout)...")

            with sinergym_workdir():
                env = gymnasium.make(cfg["env_name"])
            raw_obs, _ = env.reset(seed=self.seed)
            obs = self._normalize(raw_obs)
            done = False
            ep_reward = 0.0

            while not done:
                global_step += 1
                if global_step < 500:
                    action = env.action_space.sample()
                else:
                    action = dyna.select_action(obs)

                raw_next, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_obs = self._normalize(raw_next)

                if global_step >= 500:
                    dyna.train_step(
                        obs,
                        action,
                        float(reward),
                        next_obs,
                        done,
                        bid,
                        global_step,
                    )
                else:
                    dyna.buffer.add_real(obs, action, float(reward), next_obs, done)

                ep_reward += float(reward)
                obs = next_obs

            env.close()
            logger.info(f"  {bid}: reward={ep_reward:.1f}, steps={global_step}")
            dyna.on_episode_end()

        return dyna

    def _collect_target_data(self) -> dict[str, np.ndarray]:
        """Collect full episode from target building."""
        env_name = self.target_config["env_name"]
        with sinergym_workdir():
            env = gymnasium.make(env_name)
        raw_obs, _ = env.reset(seed=self.seed + 100)

        raw_states, raw_next_states = [], []
        states, actions, next_states, rewards_list = [], [], [], []
        done = False
        while not done:
            action = env.action_space.sample()
            raw_next, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            raw_states.append(raw_obs.copy())
            raw_next_states.append(raw_next.copy())
            states.append(self._normalize(raw_obs))
            actions.append(action)
            next_states.append(self._normalize(raw_next))
            rewards_list.append(float(reward))
            raw_obs = raw_next

        env.close()
        logger.info(f"Target data: {len(states)} steps")

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "rewards": np.array(rewards_list, dtype=np.float32),
            "raw_states": np.array(raw_states, dtype=np.float32),
            "raw_next_states": np.array(raw_next_states, dtype=np.float32),
        }

    def _run_transfer(
        self, source_dyna: DynaSAC, target_data: dict, n_adapt_steps: int
    ) -> float:
        """Transfer: fine-tune adapter + SAC on limited target data."""
        # Deep copy so each data budget starts fresh from source
        dyna = self._clone_dyna(source_dyna)

        # Reset optimizer states (prevent source Adam momentum from leaking)
        dyna.sac_trainer.actor_optimizer = torch.optim.Adam(
            dyna.actor.parameters(), lr=self.config.sac.actor_lr
        )
        dyna.sac_trainer.critic_optimizer = torch.optim.Adam(
            dyna.critic.parameters(), lr=self.config.sac.critic_lr
        )
        if dyna.sac_trainer.autotune_alpha:
            dyna.sac_trainer.alpha_optimizer = torch.optim.Adam(
                [dyna.sac_trainer.log_alpha], lr=self.config.sac.alpha_lr
            )
        # Clear buffers from source training
        dyna.buffer = type(dyna.buffer)(
            real_capacity=self.config.buffer_size,
            model_capacity=min(self.config.buffer_size, 10_000),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

        n_buildings = len(self.source_configs)
        target_idx = str(n_buildings)

        dyna.world_model.encoder.add_adapter(target_idx)
        # Warm-start adapter from first source building
        src_adapter_id = "0"
        if src_adapter_id in dyna.world_model.encoder.adapters:
            src_state = dyna.world_model.encoder.adapters[src_adapter_id].state_dict()
            dyna.world_model.encoder.adapters[target_idx].load_state_dict(src_state)
        dyna.world_model.encoder.to(self.device)

        # Step 1: Fine-tune WM adapter on limited data
        for param in dyna.world_model.parameters():
            param.requires_grad_(False)
        for param in dyna.world_model.encoder.adapters[target_idx].parameters():
            param.requires_grad_(True)

        # Uniformly sample across the full episode (not just first N steps)
        # This ensures adaptation data covers all seasons
        total = len(target_data["states"])
        indices = np.linspace(0, total - 1, n_adapt_steps, dtype=int)
        adapt_s = torch.tensor(target_data["states"][indices], device=self.device)
        adapt_a = torch.tensor(target_data["actions"][indices], device=self.device)
        adapt_sn = torch.tensor(target_data["next_states"][indices], device=self.device)

        optimizer = torch.optim.Adam(
            dyna.world_model.encoder.adapters[target_idx].parameters(),
            lr=1e-3,
        )
        for _epoch in range(50):
            dyna.world_model.train()
            loss, _ = dyna.world_model.compute_loss(
                adapt_s,
                adapt_a,
                adapt_sn,
                building_id=target_idx,
                sparsity_lambda=0.1,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dyna.world_model.normalize_atoms()

        for param in dyna.world_model.parameters():
            param.requires_grad_(True)

        # Step 2: Fill buffer with same sampled data
        for i in indices:
            dyna.buffer.add_real(
                target_data["states"][i],
                target_data["actions"][i],
                target_data["rewards"][i],
                target_data["next_states"][i],
                False,
            )

        # SAC updates using real target data + model rollouts
        batch_size = min(64, n_adapt_steps)
        n_sac_updates = max(200, 200 * n_adapt_steps // 96)  # scale with data
        for step in range(n_sac_updates):
            # Generate rollouts from adapted WM
            if n_adapt_steps >= batch_size and step >= 50:
                start = dyna.buffer.real_buffer.sample(5, self.device)
                rollout = dyna.rollout_gen.generate(
                    start["states"].cpu().numpy(), target_idx, horizon=1
                )
                dyna.buffer.add_model_batch(
                    rollout["states"],
                    rollout["actions"],
                    rollout["rewards"],
                    rollout["next_states"],
                    rollout["dones"],
                )

            # SAC update (use mixed buffer: real + model data)
            ratio = self.config.dyna.model_to_real_ratio if step >= 50 else 0.0
            batch = dyna.buffer.sample(
                batch_size, model_ratio=ratio, device=self.device
            )
            dyna.sac_trainer.update(
                batch["states"],
                batch["actions"],
                batch["rewards"],
                batch["next_states"],
                batch["dones"],
            )

        eval_reward = self._evaluate_on_target(dyna, target_idx)
        return eval_reward

    def _clone_dyna(self, source: DynaSAC) -> DynaSAC:
        """Create a fresh DynaSAC and load source weights (avoids deepcopy issues)."""
        clone = DynaSAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            building_ids=source.building_ids,
            dictionary=self.dictionary,
            config=source.config,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            obs_mean=self._obs_mean_t,
            obs_std=self._obs_std_t,
        )
        clone.world_model.load_state_dict(source.world_model.state_dict())
        clone.actor.load_state_dict(source.actor.state_dict())
        clone.critic.load_state_dict(source.critic.state_dict())
        clone.critic_target.load_state_dict(source.critic_target.state_dict())
        return clone

    def _run_context_transfer(
        self, source_dyna: DynaSAC, target_data: dict, n_adapt_steps: int
    ) -> float:
        """Context-conditioned transfer: inherit source WM, fine-tune on target."""
        dyna = self._clone_dyna(source_dyna)
        # dyna.world_model is the SOURCE-TRAINED ContextDynamicsModel
        # (context_encoder + conditioned_encoder + dictionary all trained on source)
        ctx_model = dyna.world_model

        # Build context from ALL sampled target transitions (not just first K)
        total = len(target_data["states"])
        indices = np.linspace(0, total - 1, n_adapt_steps, dtype=int)

        s_ctx = target_data["states"][indices]
        a_ctx = target_data["actions"][indices]
        sn_ctx = target_data["next_states"][indices]
        delta_ctx = sn_ctx - s_ctx

        # Build transition tensor: (1, N, 2*d + m) — use all available data
        transitions = np.concatenate([s_ctx, a_ctx, delta_ctx], axis=-1)
        transitions_t = torch.tensor(
            transitions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Infer context (zero-shot, mean-pooled over all transitions)
        with torch.no_grad():
            target_context = ctx_model.infer_context(transitions_t)  # (1, context_dim)

        # Optional: fine-tune ONLY context encoder on target data
        # Dictionary and conditioned encoder are frozen to preserve source knowledge
        if n_adapt_steps >= 50:
            adapt_s = torch.tensor(target_data["states"][indices], device=self.device)
            adapt_a = torch.tensor(target_data["actions"][indices], device=self.device)
            adapt_sn = torch.tensor(
                target_data["next_states"][indices], device=self.device
            )
            context_expanded = target_context.expand(len(indices), -1)

            # Freeze dictionary only (preserve source thermal dynamics)
            # Both encoders (context + conditioned) are jointly trainable
            if isinstance(ctx_model.dictionary, torch.nn.Parameter):
                ctx_model.dictionary.requires_grad_(False)

            encoder_params = list(ctx_model.context_encoder.parameters()) + list(
                ctx_model.encoder.parameters()
            )
            optimizer = torch.optim.Adam(encoder_params, lr=1e-3)
            for _epoch in range(20):
                ctx_model.train()
                # Re-infer context each epoch (encoder improves)
                target_context = ctx_model.infer_context(transitions_t)
                context_expanded = target_context.expand(len(indices), -1)
                loss, _ = ctx_model.compute_loss(
                    adapt_s,
                    adapt_a,
                    adapt_sn,
                    context=context_expanded,
                    sparsity_lambda=0.1,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Restore dictionary grad flag
            if isinstance(ctx_model.dictionary, torch.nn.Parameter):
                ctx_model.dictionary.requires_grad_(True)

            # Final context after fine-tuning
            with torch.no_grad():
                target_context = ctx_model.infer_context(transitions_t)

        # Fill buffer with sampled target data
        for i in indices:
            dyna.buffer.add_real(
                target_data["states"][i],
                target_data["actions"][i],
                target_data["rewards"][i],
                target_data["next_states"][i],
                False,
            )

        # SAC updates with context-conditioned rollouts
        # Fixed 200 updates: more data improves batch diversity, not update count
        batch_size = min(64, n_adapt_steps)
        n_sac_updates = 200
        for step in range(n_sac_updates):
            if n_adapt_steps >= batch_size and step >= 50:
                start = dyna.buffer.real_buffer.sample(5, self.device)
                rollout = dyna.rollout_gen.generate(
                    start["states"].cpu().numpy(),
                    horizon=1,
                    context=target_context,
                )
                dyna.buffer.add_model_batch(
                    rollout["states"],
                    rollout["actions"],
                    rollout["rewards"],
                    rollout["next_states"],
                    rollout["dones"],
                )

            ratio = self.config.dyna.model_to_real_ratio if step >= 50 else 0.0
            batch = dyna.buffer.sample(
                batch_size, model_ratio=ratio, device=self.device
            )
            dyna.sac_trainer.update(
                batch["states"],
                batch["actions"],
                batch["rewards"],
                batch["next_states"],
                batch["dones"],
            )

        eval_reward = self._evaluate_on_target(dyna, "0")
        return eval_reward

    def _run_context_transfer_no_rollout(
        self, source_dyna: DynaSAC, target_data: dict, n_adapt_steps: int
    ) -> float:
        """Context transfer WITHOUT model rollouts (ablation B).

        Same as _run_context_transfer but SAC trains on real data only.
        Isolates policy-transfer contribution from WM-rollout contribution.
        """
        dyna = self._clone_dyna(source_dyna)
        ctx_model = dyna.world_model

        # Context inference + fine-tune (same as full transfer)
        total = len(target_data["states"])
        indices = np.linspace(0, total - 1, n_adapt_steps, dtype=int)

        s_ctx = target_data["states"][indices]
        a_ctx = target_data["actions"][indices]
        sn_ctx = target_data["next_states"][indices]
        delta_ctx = sn_ctx - s_ctx
        transitions = np.concatenate([s_ctx, a_ctx, delta_ctx], axis=-1)
        transitions_t = torch.tensor(
            transitions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            target_context = ctx_model.infer_context(transitions_t)

        if n_adapt_steps >= 50:
            adapt_s = torch.tensor(target_data["states"][indices], device=self.device)
            adapt_a = torch.tensor(target_data["actions"][indices], device=self.device)
            adapt_sn = torch.tensor(
                target_data["next_states"][indices], device=self.device
            )
            context_expanded = target_context.expand(len(indices), -1)
            if isinstance(ctx_model.dictionary, torch.nn.Parameter):
                ctx_model.dictionary.requires_grad_(False)
            encoder_params = list(ctx_model.context_encoder.parameters()) + list(
                ctx_model.encoder.parameters()
            )
            optimizer = torch.optim.Adam(encoder_params, lr=1e-3)
            for _epoch in range(20):
                ctx_model.train()
                target_context = ctx_model.infer_context(transitions_t)
                context_expanded = target_context.expand(len(indices), -1)
                loss, _ = ctx_model.compute_loss(
                    adapt_s,
                    adapt_a,
                    adapt_sn,
                    context=context_expanded,
                    sparsity_lambda=0.1,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if isinstance(ctx_model.dictionary, torch.nn.Parameter):
                ctx_model.dictionary.requires_grad_(True)
            with torch.no_grad():
                target_context = ctx_model.infer_context(transitions_t)

        # Fill buffer with sampled target data
        for i in indices:
            dyna.buffer.add_real(
                target_data["states"][i],
                target_data["actions"][i],
                target_data["rewards"][i],
                target_data["next_states"][i],
                False,
            )

        # SAC updates WITHOUT rollouts — real data only
        batch_size = min(64, n_adapt_steps)
        n_sac_updates = 200
        for _step in range(n_sac_updates):
            batch = dyna.buffer.real_buffer.sample(batch_size, self.device)
            dyna.sac_trainer.update(
                batch["states"],
                batch["actions"],
                batch["rewards"],
                batch["next_states"],
                batch["dones"],
            )

        eval_reward = self._evaluate_on_target(dyna, "0")
        return eval_reward

    def _run_from_scratch(self, target_data: dict, n_steps: int) -> float:
        """Train from scratch with only n_steps of target data."""
        # Compute obs stats from target data itself (no source leakage)
        target_obs_mean = torch.tensor(
            target_data["raw_states"].mean(axis=0), dtype=torch.float32
        )
        target_obs_std = torch.tensor(
            np.maximum(target_data["raw_states"].std(axis=0), 1e-8),
            dtype=torch.float32,
        )

        # Fresh Dyna-SAC with random dictionary (true scratch, dict mode)
        random_dict = torch.randn_like(self.dictionary)
        random_dict = random_dict / random_dict.norm(dim=0, keepdim=True)

        # Scratch baseline uses dict mode (no context, no transfer knowledge)
        scratch_config = self.config.model_copy(update={"mode": "dict"})
        dyna = DynaSAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            building_ids=["scratch"],
            dictionary=random_dict,
            config=scratch_config,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            obs_mean=target_obs_mean,
            obs_std=target_obs_std,
        )

        # Train world model on uniformly sampled data (same as transfer)
        total = len(target_data["states"])
        indices = np.linspace(0, total - 1, n_steps, dtype=int)
        s = torch.tensor(target_data["states"][indices], device=self.device)
        a = torch.tensor(target_data["actions"][indices], device=self.device)
        sn = torch.tensor(target_data["next_states"][indices], device=self.device)

        for _epoch in range(50):
            dyna.world_model.train()
            loss, _ = dyna.world_model.compute_loss(
                s,
                a,
                sn,
                building_id="0",
                sparsity_lambda=0.1,
            )
            dyna.wm_trainer.optimizer.zero_grad()
            loss.backward()
            dyna.wm_trainer.optimizer.step()
            dyna.world_model.normalize_atoms()

        # Fill buffer with same sampled data
        for i in indices:
            dyna.buffer.add_real(
                target_data["states"][i],
                target_data["actions"][i],
                target_data["rewards"][i],
                target_data["next_states"][i],
                False,
            )

        # SAC updates with rollouts (SAME as transfer for fairness)
        # Fixed 200 updates: more data improves batch diversity, not update count
        batch_size = min(64, n_steps)
        n_sac_updates = 200
        for step in range(n_sac_updates):
            # Generate rollouts (same as transfer)
            if n_steps >= batch_size and step >= 50:
                start = dyna.buffer.real_buffer.sample(5, self.device)
                rollout = dyna.rollout_gen.generate(
                    start["states"].cpu().numpy(), "0", horizon=1
                )
                dyna.buffer.add_model_batch(
                    rollout["states"],
                    rollout["actions"],
                    rollout["rewards"],
                    rollout["next_states"],
                    rollout["dones"],
                )

            # SAC update (mixed buffer, same as transfer)
            ratio = self.config.dyna.model_to_real_ratio if step >= 50 else 0.0
            batch = dyna.buffer.sample(
                batch_size, model_ratio=ratio, device=self.device
            )
            dyna.sac_trainer.update(
                batch["states"],
                batch["actions"],
                batch["rewards"],
                batch["next_states"],
                batch["dones"],
            )

        eval_reward = self._evaluate_on_target(dyna, "0")
        return eval_reward

    def _evaluate_on_target(self, dyna: DynaSAC, building_id: str) -> float:
        """Evaluate policy on target building (full episode)."""
        env_name = self.target_config["env_name"]
        with sinergym_workdir():
            env = gymnasium.make(env_name)
        raw_obs, _ = env.reset(seed=self.seed + 200)
        obs = self._normalize(raw_obs)

        total_reward = 0.0
        done = False
        while not done:
            action = dyna.select_action(obs, deterministic=True)
            raw_next, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            obs = self._normalize(raw_next)
            total_reward += float(reward)

        env.close()
        return total_reward
