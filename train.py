import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import wandb

from data.data_loader import get_dataloaders
from models.base import BaseModel
from run.trainer import trainer
from utils.utils import *


def main():
    assert torch.cuda.is_available(), "You need cuda to train"

    # Load configurations
    parser = argparse.ArgumentParser(
        description="Speech-Driven 3D Facial Animation with A-V Guidance"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to train the model",
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    parser.add_argument(
        "--save_root_dir", type=str, default="outputs",
    )
    parser.add_argument(
        "--exp_name", type=str,
    )
    cmd_input = parser.parse_args()

    if cmd_input.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif cmd_input.dataset == "BIWI":
        args = load_config("config/biwi.yaml")

    if cmd_input.exp_name is not None:
        args.exp_name = cmd_input.exp_name
    else:
        if args.exp_name is None:
            import datetime
            args.exp_name = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

    # Set the paths
    if cmd_input.save_root_dir is not None and os.path.exists(cmd_input.save_root_dir):
        args.save_root_dir = cmd_input.save_root_dir
    args.save_root_dir = os.path.join(args.save_root_dir, args.exp_name)
    args.save_model_path = os.path.join(args.save_root_dir, args.save_model_path)

    # Make directories to save results
    make_dirs(args.save_model_path)
    print(args.save_model_path)

    if cmd_input.debug:
        args.log_wandb = False

    if args.log_wandb:
        import datetime
        exp_name = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S") # e.g., 24_02_16-17_26_20

        wandb.login()
        wandb.init( project=args.wandb_project, name=exp_name, config=vars(args))

    # Build model - facial animator & lip reader
    model = BaseModel(args)

    # Load data
    dataset = get_dataloaders(args)

    # Train the model
    trainer(args, dataset, model)


if __name__=="__main__":
    main()