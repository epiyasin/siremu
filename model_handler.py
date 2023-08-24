import os
import torch
import torch.nn as nn
import torch.optim as optim
from training import train_model
from utils import select_model, check_model_exists
from plotting import plot_losses
from settings import Settings

settings = Settings("config.json")

def initialize_model(settings):
    return select_model(settings)

def load_pretrained_model(model, model_type, source):
    sub_folder = f"{source}_models"
    model_path = os.path.join("cached_models", sub_folder, f"{source}_{model_type}model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model


def train_and_save_model(model, train_loader, val_loader, settings):
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.lr_scheduler['learning_rate'])

    # Train the model
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, settings)
    plot_losses(train_losses, val_losses)
    
    # Get the sub-folder based on the source
    sub_folder = f"{settings.execution_source}_models"
    model_folder = os.path.join("cached_models", sub_folder)
    
    # Check and create the sub-folder if not exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_folder, f"{settings.execution_source}_{settings.model_type}model.pth"))
    return model

def handle_model(train_loader, val_loader, settings):
    model = initialize_model(settings)

    # Check and create the parent folder if not exists
    if not os.path.exists("cached_models"):
        os.makedirs("cached_models")

    # Get the sub-folder based on the source
    sub_folder = f"{settings.execution_source}_models"
    model_folder = os.path.join("cached_models", sub_folder)
    
    # Check and create the sub-folder if not exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_exists = check_model_exists(settings.model_type, settings.execution_source)

    if model_exists and settings.cached_model:
        model = load_pretrained_model(model, settings.model_type, settings.execution_source)
    else:
        if not model_exists:
            print("No saved models present, running training...")
        model = train_and_save_model(model, train_loader, val_loader, settings)

    return model

