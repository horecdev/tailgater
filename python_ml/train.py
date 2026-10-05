import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

from siamfc import SiamFC
from dataset import SiamFCDataset
from loss import SiamFCLoss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    processed_dir = Path("GOT-10k-Processed")
    save_dir = Path("train")
    save_dir.mkdir(exist_ok=True)

    batch_size = 32
    epochs = 500
    lr = 1e-3

    dataset = SiamFCDataset(processed_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SiamFC().to(device)
    loss_fn = SiamFCLoss(radius=16).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    print_every = 40

    history = {"epoch_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        
        for step, (exemplar, search, shift_x, shift_y) in enumerate(train_loader):
            exemplar = exemplar.to(device)
            search = search.to(device)
            shift_x = shift_x.to(device)
            shift_y = shift_y.to(device)
            
            optimizer.zero_grad()
            
            pred_heatmap = model(search, exemplar)
            loss = loss_fn(pred_heatmap, shift_x, shift_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            if (step) % print_every == 0:
                print(f"EPOCH: [{epoch}/{epochs}] STEP: [{step+1}/{len(train_loader)}] LOSS: {loss.item():.4f}")

        print(f"Scheduler LR: {scheduler.get_last_lr()}")
        scheduler.step()
        epoch_loss = total_epoch_loss / len(train_loader)
        history["epoch_loss"].append(epoch_loss)
        
        print(f"Finished epoch {epoch}. | Average loss: {epoch_loss:.4f}")
        
        torch.save(model.state_dict(), save_dir / f"siamfc_epoch_{epoch}.pth")
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f)
    
if __name__ == "__main__":
    train()
        