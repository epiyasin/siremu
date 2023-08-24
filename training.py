import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def train_model(model, criterion, optimizer, train_loader, val_loader, settings):
    # Retrieve values from settings
    epochs = settings.neural_net.nn_epochs
    step_size = settings.neural_net.lr_scheduler.step_size
    gamma = settings.neural_net.lr_scheduler.gamma

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    progress_bar = tqdm(range(epochs), desc='Training model', dynamic_ncols=True)

    train_losses = []
    val_losses = []

    for epoch in progress_bar:
        total_train_loss = 0
        total_train_batches = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_batches += 1
        avg_train_loss = total_train_loss / total_train_batches
        train_losses.append(avg_train_loss)
        scheduler.step()

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
                val_losses.append(avg_val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_description(f'Training model (Loss: {avg_train_loss:.2f}, Avg Val Loss: {avg_val_loss:.2f}, LR: {current_lr:.2e}), completed')
    
    return train_losses, val_losses