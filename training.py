import torch
from tqdm import tqdm

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    progress_bar = tqdm(range(epochs), desc='Training model', dynamic_ncols=True)

    for epoch in progress_bar:
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        if val_loader:
            with torch.no_grad():
                val_loss = 0
                val_batches = 0
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    val_batches += 1
                avg_val_loss = val_loss / val_batches
                progress_bar.set_description(f'Training model (Loss: {loss.item():.4f}, Avg Val Loss: {avg_val_loss:.4f}), completed')
