import torch
from tqdm import tqdm

def train_one_epoch(model, train_dataloader, optimizer, device, cross_entropy_loss, reg_loss, neg_loss):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_dataloader), unit='batch') as pbar:
        for data in train_dataloader:
            spec, image, heatmap, diff, height, detect, cls, traj = (d.to(device) for d in data)
            optimizer.zero_grad()
            mask_heat = (torch.max(heatmap.view(heatmap.size(0), -1), dim=1).values != 0).float().view(-1, 1, 1)
            mask_traj = (traj != 0).float()
            mask_z = (height != 0).float()
            mask_off = (diff != 0).float()
            h, z, o, d, t, c = model(spec, image)
            loss_cls = cross_entropy_loss(c, cls)
            loss_detect = cross_entropy_loss(d, detect)
            loss_traj = reg_loss(t * mask_traj, traj * mask_traj)
            loss_z = reg_loss(z * mask_z, height * mask_z)
            loss_heatmap = neg_loss(h * mask_heat, heatmap * mask_heat)
            loss_off = reg_loss(o * mask_off, diff * mask_off)
            loss = loss_heatmap + loss_z + loss_off * 0.5 + loss_detect * 0.3 + loss_traj * 0.3 + loss_cls * 0.3
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    return total_loss / len(train_dataloader)


def validate_one_epoch(model, val_dataloader, device, cross_entropy_loss, reg_loss, neg_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_dataloader:
            spec, image, heatmap, diff, height, detect, cls, traj = (d.to(device) for d in data)
            mask_heat = (torch.max(heatmap.view(heatmap.size(0), -1), dim=1).values != 0).float().view(-1, 1, 1)
            mask_traj = (traj != 0).float()
            mask_z = (height != 0).float()
            mask_off = (diff != 0).float()
            h, z, o, d, t, c = model(spec, image)
            loss_cls = cross_entropy_loss(c, cls)
            loss_detect = cross_entropy_loss(d, detect)
            loss_traj = reg_loss(t * mask_traj, traj * mask_traj)
            loss_z = reg_loss(z * mask_z, height * mask_z)
            loss_heatmap = neg_loss(h * mask_heat, heatmap * mask_heat)
            loss_off = reg_loss(o * mask_off, diff * mask_off)
            loss = loss_heatmap + loss_z + loss_off * 0.5 + loss_detect * 0.3 + loss_traj * 0.3 + loss_cls * 0.3
            total_loss += loss.item()

    return total_loss / len(val_dataloader)


def train_and_validate(model, train_dataloader, val_dataloader, optimizer, device, cross_entropy_loss, reg_loss, neg_loss, epochs, save_path):
    """
    Train and validate the model.

    Args:
        model: The PyTorch model.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        optimizer: The optimizer.
        device: The device (CPU or GPU).
        cross_entropy_loss: Loss function for classification.
        reg_loss: Loss used for regression task
        neg_loss: Negative loss function for heatmap optimization
        epochs: Number of epochs to train.
        save_path: Path to save the model weights.
    """
    val_min = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, cross_entropy_loss, reg_loss, neg_loss)
        print(f"Train Loss: {train_loss:.4f}")
        if epoch%10==0:
            val_loss = validate_one_epoch(model, val_dataloader, device, cross_entropy_loss, reg_loss, neg_loss)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < val_min:
                val_min = val_loss
                torch.save(model.state_dict(), f'{save_path}/best_epoch.pth')

            torch.save(model.state_dict(), f'{save_path}/last_epoch.pth')
