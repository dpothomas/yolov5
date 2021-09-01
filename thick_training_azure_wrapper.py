import sys
from pathlib import Path
import yaml
import os
from pprint import pprint
import subprocess
# Code in submitted run
try:
    from azureml.core import Run
except:
    pass


def _setup():
    # Setting up W&B
    if os.getenv(
            'AZUREML_ARM_RESOURCEGROUP') is not None:  # checking if we are in Azure, unless you have really weird local env variables
        run = Run.get_context()
        secret_value = run.get_secret(name="WANDB-BOT-API-KEY")  # Secret called
        # WANDB-API-KEY2 was created in Azure KeyVault
        os.environ['WANDB_API_KEY'] = secret_value

    # subprocess.run(["ls"])
    # subprocess.run(["pip", "show", "wandb"])
    # subprocess.run(["pip", "install", "wandb==0.11.2"])
    # subprocess.run(["pip", "show", "wandb"])
    subprocess.run(["pip", "install", "typer"])
    subprocess.run(["pip", "install", "aisa_utils-1.0.1-py3-none-any.whl"])


_setup()
import typer
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
    test_dataset_location: Path = typer.Argument(
        ...,
        help="Location of test dataset (yaml or path to root).",
    ),
    test_video_dataset_location: Path = typer.Argument(
        ...,
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
):
    import train
    import mojo_test

    #### TRAINING CODE ####
    # Create data.yaml is a root path is given (hard code extra values for now).
    if train_dataset_location.is_dir():
        train_data = dict(
            path=str(train_dataset_location.as_posix()),
            train="images/train",
            val="images/val",
            nc=1,
            names=["Sperm"],
        )
        pprint(train_data)
        train_yaml_file_path = Path("data.yaml")
        with train_yaml_file_path.open("w") as file:
            yaml.dump(train_data, file)
        print(f"Created data yaml at {train_yaml_file_path} containing:")
    elif train_dataset_location.is_file() and train_dataset_location.suffix == ".yaml":
        train_yaml_file_path = train_dataset_location
    else:
        raise Exception(f"{train_dataset_location} not supported as an dataset type.")

    print("Running training function...")
    hyp_path = Path("mojo_hyp.yaml")
    hyp_path = None
    if hyp_path is not None and hyp_path.is_file():
        path_to_best_model = train.run(
            cfg=f"models/{yolo_model_version}.yaml",
            data=f"{train_yaml_file_path}",
            hyp=f"{hyp_path}",
            project="test_training_results",
            name=f"{yolo_model_version}-{image_size}-{hyp_path.stem}",
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            workers=4,
            entity="mojo-ia"
        )
    else:
        path_to_best_model = train.run(
            cfg=f"models/{yolo_model_version}.yaml",
            data=f"{train_yaml_file_path}",
            # f"--hyp {root / 'hyp' / 'new_current.yaml'} "
            project="test_training_results",
            name=f"{yolo_model_version}-{image_size}",
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            workers=4,
            entity="mojo-ia"
        )
    print("Finished training function...")
    #### END OF TRAINING CODE ####

    #### TESTING ####
    test_data_root_dir = Path(sys.argv[3])
    data_root_dir_test_videos = Path(sys.argv[4])

    mojo_test_data = dict(
        path=str(test_data_root_dir),
        test="images/test",
        nc=1,
        names=["Sperm"],
    )
    mojo_test_yaml_file_path = Path("test_data.yaml")
    with mojo_test_yaml_file_path.open("w") as file:
        yaml.dump(mojo_test_data, file)
    print(f"Created data yaml at {mojo_test_yaml_file_path} containing:")

    print("Running mojo testing function...")
    mojo_test.mojo_test(
        mojo_test_yaml_file_path,
        [path_to_best_model],
        batch_size=batch_size,
        imgsz=image_size,
        project="test_training_results",
        name=f"{yolo_model_version}-{image_size}",
        entity="mojo-ia",
        test_video_root=data_root_dir_test_videos
    )
    print("Finished mojo testing function...")


def cli():
    app()


if __name__ == '__main__':
    app()

