# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import math

import click
import yaml
import optuna
from functools import partial

from mbart_qe.nmt_estimator import NMTEstimator
from pytorch_lightning import seed_everything
from trainer import TrainerConfig, build_trainer
from mbart_qe.configs import ModelConfig

@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    model_config = ModelConfig(yaml_file)
    model = NMTEstimator(**model_config.to_dict())
    trainer.fit(model)


@cli.command(name="search")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
@click.option(
    "--n_trials",
    type=int,
    default=25,
    help="Number of search trials",
)
def search(config: str, n_trials: int) -> None:
    
    def objective(trial, train_config, model_config):
        model_config.nr_frozen_epochs = trial.suggest_uniform(
            "nr_frozen_epochs", 0.05, 0.99
        )
        model_config.encoder_learning_rate = trial.suggest_loguniform(
            "encoder_learning_rate", 1e-6, 9.0e-6 
        )
        model_config.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-6, 1e-5
        )
        model_config.glass_box_bottleneck = trial.suggest_categorical(
            "glass_box_bottleneck", 
            ["128", "64", "256", "512"]
        )
        model_config.glass_box_bottleneck = int(model_config.glass_box_bottleneck)

        seed_everything(train_config.seed)
        trainer = build_trainer(train_config.namespace())
        model = NMTEstimator(model_config.to_dict())
        try:
            trainer.fit(model)
        except RuntimeError:
            click.secho("CUDA OUT OF MEMORY, SKIPPING TRIAL", fg="red")
            return -1

        best_score = trainer.callbacks[0].best_score.item()
        return -1 if math.isnan(best_score) else best_score
    
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    train_config = TrainerConfig(yaml_file)
    model_config = NMTEstimator.ModelConfig(yaml_file)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    try:
        study.optimize(
            partial(objective, train_config=train_config, model_config=model_config),
            n_trials=n_trials,
        )

    except KeyboardInterrupt:
        click.secho("Early stopping search caused by ctrl-C", fg="red")

    except Exception as e:
        click.secho(
            f"Error occured during search: {e}; current best params are {study.best_params}",
            fg="red",
        )

    try:
        click.secho(
            "Number of finished trials: {}".format(len(study.trials)), fg="yellow"
        )
        click.secho("Best trial:", fg="yellow")
        trial = study.best_trial
        click.secho("  Value: {}".format(trial.value), fg="yellow")
        click.secho("  Params: ", fg="yellow")
        for key, value in trial.params.items():
            click.secho("    {}: {}".format(key, value), fg="blue")

    except Exception as e:
        click.secho(f"Logging at end of search failed: {e}", fg="red")

    click.secho(f"Saving Optuna plots for this search to experiments/", fg="yellow")
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("experiments/optimization_history.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html("experiments/parallel_coordinate.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("experiments/param_importances.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")



if __name__ == "__main__":
    cli()