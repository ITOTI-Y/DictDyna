"""DictDyna unified CLI entry point."""

from typing import Annotated

import typer
from loguru import logger
from omegaconf import OmegaConf

app = typer.Typer(
    name="dictdyna",
    help="DictDyna: Dictionary Learning based MBRL for Building Energy Management",
)
visualize_app = typer.Typer(help="Visualization commands")
app.add_typer(visualize_app, name="visualize")


def _load_config(config: str | None, overrides: list[str] | None = None) -> dict:
    """Load YAML config with optional CLI overrides."""
    cfg = OmegaConf.load(config) if config else OmegaConf.create({})

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, value)

    return OmegaConf.to_container(cfg, resolve=True)  # ty: ignore[invalid-return-type]


ConfigOption = Annotated[
    str | None, typer.Option("--config", "-c", help="Path to YAML config file")
]
OverrideOption = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Config overrides (key=value)"),
]


@app.command()
def collect(
    config: ConfigOption = None,
    override: OverrideOption = None,
    policy: Annotated[
        str, typer.Option(help="Collection policy: random or rbc")
    ] = "rbc",
    n_episodes: Annotated[int, typer.Option(help="Episodes per building")] = 2,
    env_name: Annotated[
        str | None,
        typer.Option("--env", help="Single env name (overrides config)"),
    ] = None,
    building_id: Annotated[
        str | None,
        typer.Option("--bid", help="Building ID (with --env)"),
    ] = None,
) -> None:
    """Phase 0: Collect offline data from Sinergym environments."""
    cfg = _load_config(config, override)
    logger.info("Starting offline data collection")

    from src.data.offline_collector import OfflineCollector

    # Allow single env from CLI or multiple from config
    if env_name:
        buildings = [{"env_name": env_name, "building_id": building_id or "default"}]
    else:
        buildings = cfg.get("data", {}).get("buildings", [])

    if not buildings:
        logger.error("No buildings configured. Use --config or --env.")
        raise typer.Exit(1)

    output_dir = cfg.get("data", {}).get("output_path", "data/offline_rollouts")
    diffs_dir = "data/processed/state_diffs"

    collector = OfflineCollector(
        building_configs=buildings,
        policy=policy,
        n_episodes=n_episodes,
        output_dir=output_dir,
        diffs_dir=diffs_dir,
    )
    data = collector.collect()

    # Print data summary
    for bid, d in data.items():
        logger.info(
            f"  {bid}: states={d['states'].shape}, "
            f"diffs range=[{d['diffs'].min():.3f}, {d['diffs'].max():.3f}]"
        )
    logger.info("Data collection complete")


@app.command()
def pretrain(
    config: ConfigOption = None,
    override: OverrideOption = None,
    method: Annotated[
        str, typer.Option(help="Dictionary learning method: ksvd or online")
    ] = "ksvd",
) -> None:
    """Phase I: Pretrain dictionary on offline state diffs."""
    cfg = _load_config(config, override)
    logger.info("Starting dictionary pretraining")

    from src.dictionary.pretrain import pretrain_dictionary

    dict_cfg = cfg.get("dictionary", {})
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})

    pretrain_dictionary(
        data_dir=data_cfg.get("output_path", "data/processed/state_diffs"),
        n_atoms=dict_cfg.get("n_atoms", 128),
        method=method,
        max_iter=dict_cfg.get("pretrain_epochs", 100),
        output_path=output_cfg.get("dict_path", "output/pretrained/dict.pt"),
    )
    logger.info("Dictionary pretraining complete")


@app.command()
def train(
    config: ConfigOption = None,
    override: OverrideOption = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    env_name: Annotated[
        str, typer.Option("--env", help="Sinergym environment name")
    ] = "Eplus-5zone-hot-continuous-v1",
    building_id: Annotated[
        str, typer.Option("--bid", help="Building identifier")
    ] = "office_hot",
    total_timesteps: Annotated[
        int, typer.Option("--steps", help="Total training timesteps")
    ] = 26280,
    dict_path: Annotated[
        str, typer.Option("--dict", help="Pretrained dictionary path")
    ] = "output/pretrained/dict_k128.pt",
    wandb_enabled: Annotated[
        bool, typer.Option("--wandb/--no-wandb", help="Enable W&B logging")
    ] = False,
) -> None:
    """Phase II: Train Dyna-SAC with dictionary world model."""
    cfg = _load_config(config, override)
    logger.info(f"Starting Dyna-SAC training (seed={seed})")

    from src.agent.dyna_trainer import DynaSACTrainer
    from src.schemas import TrainSchema

    # Build TrainSchema from config + CLI overrides
    train_fields = {}
    for k, v in cfg.items():
        if k in TrainSchema.model_fields:
            train_fields[k] = v
    train_fields["seed"] = seed
    train_fields["total_timesteps"] = total_timesteps
    train_fields["device"] = cfg.get("device", "auto")
    train_cfg = TrainSchema(**train_fields)

    wandb_project = "dictdyna" if wandb_enabled else None

    trainer = DynaSACTrainer(
        env_name=env_name,
        building_id=building_id,
        dict_path=dict_path,
        config=train_cfg,
        seed=seed,
        save_dir=f"output/results/dyna_sac/{env_name}_s{seed}",
        wandb_project=wandb_project,
    )
    result = trainer.train()
    n_episodes = len(result["episode_rewards"])
    logger.info(f"Dyna-SAC training done: {n_episodes} episodes completed")


@app.command()
def transfer(
    config: ConfigOption = None,
    override: OverrideOption = None,
    days: Annotated[int, typer.Option("-d", help="Adaptation days")] = 7,
) -> None:
    """Phase III: Few-shot transfer to new building."""
    _cfg = _load_config(config, override)
    logger.info(f"Starting few-shot transfer ({days} days adaptation)")

    # TODO: Load checkpoint, create adapter, run adaptation
    logger.warning("Transfer requires trained checkpoint. Run 'train' first.")


@app.command()
def baseline(
    config: ConfigOption = None,
    override: OverrideOption = None,
    method: Annotated[
        str, typer.Option("-m", help="Baseline method: rbc, sac")
    ] = "sac",
    env_name: Annotated[
        str, typer.Option("--env", help="Sinergym environment name")
    ] = "Eplus-5zone-hot-continuous-v1",
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    total_timesteps: Annotated[
        int, typer.Option("--steps", help="Total training timesteps (SAC)")
    ] = 26280,
    n_episodes: Annotated[int, typer.Option(help="Evaluation episodes (RBC)")] = 3,
    wandb_enabled: Annotated[
        bool, typer.Option("--wandb/--no-wandb", help="Enable W&B logging")
    ] = False,
) -> None:
    """Run baseline methods (RBC, SAC) for comparison."""
    cfg = _load_config(config, override)
    logger.info(f"Running baseline: {method} on {env_name}")

    from src.agent.baseline_sac import RBCBaseline, SACBaselineTrainer

    wandb_project = "dictdyna" if wandb_enabled else None

    if method == "rbc":
        runner = RBCBaseline(
            env_name=env_name,
            n_episodes=n_episodes,
            seed=seed,
            save_dir=f"output/results/baseline_rbc/{env_name}",
        )
        result = runner.evaluate()
        logger.info(
            f"RBC result: {result['mean_reward']:.1f} ± {result['std_reward']:.1f}"
        )

    elif method == "sac":
        sac_cfg = cfg.get("sac", {})
        runner = SACBaselineTrainer(
            env_name=env_name,
            seed=seed,
            total_timesteps=total_timesteps,
            batch_size=cfg.get("training", {}).get("batch_size", 256),
            buffer_size=cfg.get("training", {}).get("buffer_size", 100_000),
            hidden_dims=sac_cfg.get("hidden_dims", [256, 256]),
            gamma=cfg.get("training", {}).get("gamma", 0.99),
            eval_freq=cfg.get("training", {}).get("eval_freq", 8760),
            save_dir=f"output/results/baseline_sac/{env_name}_s{seed}",
            device=cfg.get("device", "auto"),
            wandb_project=wandb_project,
        )
        result = runner.train()
        n_episodes_done = len(result["episode_rewards"])
        logger.info(f"SAC training done: {n_episodes_done} episodes completed")

    else:
        logger.error(f"Unknown baseline method: {method}. Use 'rbc' or 'sac'.")
        raise typer.Exit(1)


@app.command()
def ablation(
    config: ConfigOption = None,
    override: OverrideOption = None,
    all_ablations: Annotated[
        bool, typer.Option("--all", help="Run all ablations")
    ] = False,
    seeds: Annotated[int, typer.Option(help="Number of random seeds")] = 5,
) -> None:
    """Run ablation experiments."""
    _cfg = _load_config(config, override)
    logger.info(f"Running ablation experiments (seeds={seeds})")

    # TODO: Implement ablation runner
    logger.warning("Ablation experiments not yet implemented")


@app.command()
def citylearn(
    config: ConfigOption = None,
    override: OverrideOption = None,
) -> None:
    """Run CityLearn supplementary experiments."""
    _cfg = _load_config(config, override)
    logger.info("Running CityLearn experiments")

    # TODO: Implement CityLearn experiment runner
    logger.warning("CityLearn experiments not yet implemented")


@app.command()
def evaluate(
    config: ConfigOption = None,
    override: OverrideOption = None,
    checkpoint: Annotated[
        str, typer.Option(help="Path to checkpoint")
    ] = "output/checkpoints/best.pt",
    n_episodes: Annotated[int, typer.Option(help="Number of evaluation episodes")] = 3,
) -> None:
    """Evaluate a trained agent."""
    _cfg = _load_config(config, override)
    logger.info(f"Evaluating checkpoint: {checkpoint}")

    # TODO: Load agent, run evaluation
    logger.warning("Evaluation requires trained checkpoint")


@visualize_app.command("atoms")
def visualize_atoms(
    dict_path: Annotated[
        str, typer.Argument(help="Path to dictionary .pt file")
    ] = "output/pretrained/dict_k128.pt",
    output: Annotated[
        str, typer.Option(help="Output figure path")
    ] = "output/figures/atoms.png",
) -> None:
    """Visualize dictionary atoms."""
    logger.info(f"Visualizing atoms from {dict_path}")

    import torch

    dict_data = torch.load(dict_path, weights_only=False)
    dictionary = dict_data["dictionary"]
    logger.info(f"Dictionary shape: {dictionary.shape}")

    # TODO: Create atom visualization with ultraplot
    logger.warning("Atom visualization not yet implemented")


@visualize_app.command("results")
def visualize_results(
    results_dir: Annotated[
        str, typer.Argument(help="Results directory")
    ] = "output/results",
    output_dir: Annotated[
        str, typer.Option(help="Output figures directory")
    ] = "output/figures",
) -> None:
    """Generate paper figures from experiment results."""
    logger.info(f"Generating figures from {results_dir}")

    # TODO: Create all paper figures
    logger.warning("Results visualization not yet implemented")


if __name__ == "__main__":
    app()
