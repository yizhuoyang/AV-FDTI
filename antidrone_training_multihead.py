from random import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.Antidrone_dataloader_hm import AntidroneLoader
from nets.AVFDTI import AVFDTI
from utils.loss import _neg_loss
from utils.train_val import train_and_validate
from config.config import CONFIG

def main():

    config = CONFIG
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(config["annotation_lines_train"], "r") as f:
        train_lines = f.readlines()
    with open(config["annotation_lines_val"], "r") as f:
        val_lines = f.readlines()
    shuffle(train_lines)
    shuffle(val_lines)

    # Dataset and DataLoader
    train_data = AntidroneLoader(
        train_lines, config["audio_path"], config["image_path"], config["detect_path"], config["gt_path"],
        dark_aug=config["dark_aug"], audio_seq=config["audio_seq"]
    )
    val_data = AntidroneLoader(
        val_lines, config["audio_path"], config["image_path"], config["detect_path"], config["gt_path"],
        dark_aug=config["dark_aug"], audio_seq=config["audio_seq"]
    )
    train_dataloader = DataLoader(train_data, config["batchsize"], shuffle=True, num_workers=config["workers"], drop_last=True)
    val_dataloader = DataLoader(val_data, config["batchsize"], shuffle=True, num_workers=config["workers"], drop_last=True)

    # Model
    model = AVFDTI(kernel_num=config["kernel_num"],feature_dim=config["feature_dim"],dropout_rate=config["dropout_rate"],num_class=config["num_class"])
    model = model.to(device)

    if config["checkpoint_path"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999))
    mse_loss = torch.nn.MSELoss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    neg_loss = _neg_loss

    train_and_validate(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        cross_entropy_loss=cross_entropy_loss,
        reg_loss=mse_loss,
        neg_loss=neg_loss,
        epochs=config["epochs"],
        save_path=config["save_path"]
    )

if __name__ == "__main__":
    main()
