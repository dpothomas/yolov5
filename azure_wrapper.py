import logging
import tempfile
from pathlib import Path
from uuid import uuid1

import torch
import wandb
import yaml
import os
from pprint import pprint
import subprocess
import typer


logger = logging.getLogger(__name__)

# Code in submitted run
try:
    from azureml.core import Run
except:
    pass


def _find_module_wheel_path(module_name):
    module_name = module_name.replace("-", "_")
    module_wheel_paths = list(Path(".").rglob(f"**/{module_name}*.whl"))
    if len(module_wheel_paths) == 1:
        return module_wheel_paths[0]
    elif len(module_wheel_paths) == 0:
        raise Exception(f"Cannot find wheel associated with package: {module_name}")
    else:
        raise Exception(
            f"Found several wheels associated with package: {module_name} ({module_wheel_paths})"
        )


def _setup():
    # Setting up W&B
    if (
        os.getenv("AZUREML_ARM_RESOURCEGROUP") is not None
    ):  # checking if we are in Azure, unless you have really weird local env variables
        run = Run.get_context()
        secret_value = run.get_secret(name="WANDB-BOT-API-KEY")  # Secret called
        # WANDB-API-KEY2 was created in Azure KeyVault
        os.environ["WANDB_API_KEY"] = secret_value

    # Install aisa utils
    try:
        import aisa_utils
    except ModuleNotFoundError:
        aisa_utils_wheel_path = _find_module_wheel_path("aisa_utils")
        subprocess.run(["pip", "install", f"{aisa_utils_wheel_path}"])


def _get_data_yaml(dataset_location: Path, is_test: bool = False, allow_several: bool = False, *, experiment_id) -> dict:
    dataset_data_yaml_path_glob = list(dataset_location.rglob("*data.yaml"))
    if not allow_several and len(dataset_data_yaml_path_glob) > 1:
        raise Exception(
            f"Multiple data.yaml files found at {dataset_location}: {dataset_data_yaml_path_glob}"
        )
    elif len(dataset_data_yaml_path_glob) == 0:
        dataset_root = dataset_location
        data_yaml_glob = [dict(
            nc=1,
            names=["Sperm"],
            path=str(dataset_root.as_posix())
        )]

    elif len(dataset_data_yaml_path_glob) >= 1:
        data_yaml_glob = []
        for dataset_data_yaml_path in dataset_data_yaml_path_glob:
            with dataset_data_yaml_path.open("r") as file:
                data_yaml = yaml.safe_load(file)
                data_yaml["path"] = str(dataset_data_yaml_path.parent.as_posix())
                data_yaml_glob.append(data_yaml)

    for data_yaml in data_yaml_glob:
        data_yaml["experiment_id"] = experiment_id
        # Overwrite location keys to be able to launch from anywhere
        if is_test:
            data_yaml["test"] = "images/test"
            data_yaml["train"] = ""
            data_yaml["val"] = ""
        else:
            data_yaml.pop("test", None)
            data_yaml["train"] = "images/train"
            data_yaml["val"] = "images/val"

    return data_yaml_glob


_setup()
# CLI app
app = typer.Typer(add_completion=True)


@app.command()
def train(
    yolo_model_version: str = typer.Argument(
        ...,
        help="Model Name to train (ex: yolov5s)",
    ),
    train_dataset_location: Path = typer.Argument(
        ...,
        help="Location of train dataset (yaml or path to root).",
    ),
    allow_several_dataset: bool = typer.Option(
        False,
        help="Train dataset location contains several datasets ?"
    ),
    test_dataset_location: Path = typer.Argument(
        ...,
        help="Location of test dataset (yaml or path to root).",
    ),
    test_video_dataset_location: Path = typer.Option(
        None,
        help="Location of test video dataset (root of folder with videos).",
    ),
    batch_size: int = typer.Option(
        16,
        help="Size of batches to use for training.",
    ),
    image_size: int = typer.Option(
        960,
        help="Input image size.",
    ),
    epochs: int = typer.Option(
        200,
        help="Number of epochs to train for.",
    ),
    hyp: Path = typer.Option(
        Path("data/hyps/hyp.scratch.yaml"),
        help="Path to hyp file.",
    ),
    weights: Path = typer.Option(
        Path("yolov5s.pt"),
        help="Path to initial weights.",
    ),
    project: str = typer.Option(
        "test_training_results",
        help="WandB project name to upload to.",
    ),
    name: str = typer.Option(
        None,
        help="WandB run to upload to inside the project",
    ),
):
    import train
    import mojo_test

    # Wandb entity
    entity = "mojo-ia"
    # Wandb group of runs
    experiment_name = f"{name}" if name is not None else f"{yolo_model_version}-{image_size}-{hyp.stem}-{uuid1().hex[:4]}"

    logger.info("#### MOJO TRAINING BEGIN ####")
    # Create data.yaml is a root path is given (hard code extra values for now).
    if train_dataset_location.is_file() and train_dataset_location.suffix == ".yaml":
        train_data_glob = [train_dataset_location]
    elif train_dataset_location.is_dir():
        train_data_glob = _get_data_yaml(
            train_dataset_location, is_test=False, allow_several=allow_several_dataset, experiment_id=experiment_name
        )
        train_data_glob = sorted(train_data_glob, key=lambda x: x["path"], reverse=True)
    else:
        raise Exception(f"{train_dataset_location} not supported as an dataset type.")
    for dataset_idx, train_data in enumerate(train_data_glob):
        train_yaml_file_path = Path(tempfile.TemporaryDirectory().name) / "data.yaml"
        train_yaml_file_path.parent.mkdir()
        with train_yaml_file_path.open("w") as file:
            yaml.dump(train_data, file)
        logger.info(f"Created data yaml at {train_yaml_file_path} containing: {train_data}")
        pprint(train_data)

        if os.name == "nt":
            workers = 1
            cache = "disk"
        else:
            workers = 8
            cache = "ram"

        # Wandb run inside the group
        dataset_name = experiment_name + f"-{dataset_idx}"
        # Generate a random id for one run of mojo train+test
        run_id = wandb.util.generate_id()
        # Make sure not override wandb ID when pretraining
        reset_model_wandb_id(weights, run_id=run_id)
        os.environ["WANDB_RUN_ID"] = run_id
        logger.info(f"WANDB_RUN_ID injected in weights.pt and in os environment: {run_id}")

        wandb.init(id=run_id, project=project, entity=entity, name=dataset_name, resume=True, reinit=True)

        path_to_best_model = train.run(
            cfg=f"models/{yolo_model_version}.yaml",
            weights=f"{weights}",
            data=f"{train_yaml_file_path}",
            hyp=f"{hyp}",
            project=project,
            name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=image_size,
            workers=workers,
            entity=entity,
            patience=100,
            cache=cache,
        )
        logger.info("#### MOJO TRAINING END ####")

        logger.info("#### MOJO TESTING BEGIN ####")

        mojo_test_data = _get_data_yaml(
            test_dataset_location, is_test=True, allow_several=False, experiment_id=experiment_name
        )[0]
        mojo_test_yaml_file_path = Path(tempfile.TemporaryDirectory().name) / "data.yaml"
        mojo_test_yaml_file_path.parent.mkdir()
        with mojo_test_yaml_file_path.open("w") as file:
            yaml.dump(mojo_test_data, file)

        logger.info(
            f"Created data yaml at {mojo_test_yaml_file_path} containing: {mojo_test_data}"
        )

        wandb_run = wandb.init(id=run_id, project=project, entity=entity, name=dataset_name, resume=True, reinit=True)

        mojo_test.mojo_test(
            mojo_test_yaml_file_path,
            [path_to_best_model],
            batch_size=batch_size,
            imgsz=image_size,
            project=project,
            name=dataset_name,
            wandb_run=wandb_run,
            test_video_root=test_video_dataset_location,
        )
        logger.info("#### MOJO TESTING END ####")


def reset_model_wandb_id(weights, run_id=None):
    model = torch.load(weights)
    model["wandb_id"] = run_id
    torch.save(model, weights)


def cli():
    app()


if __name__ == "__main__":
    app()
