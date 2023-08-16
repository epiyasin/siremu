import os
import torch
import torch.nn as nn
import torch.optim as optim
from training import train_model
from utils import select_model, check_model_exists
from plotting import plot_losses

def initialize_model(settings):
    return select_model(settings)

def load_pretrained_model(model, model_type):
    model_path = os.path.join("cached_models", model_type + 'model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model

def train_and_save_model(model, train_loader, val_loader, settings):
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["neural_net"]["lr_scheduler"]["learning_rate"])

    # Train the model
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, settings)
    plot_losses(train_losses, val_losses)
    
    # Save the model after training
    torch.save(model.state_dict(), os.path.join("cached_models", settings["neural_net"]["model_type"] + 'model.pth'))
    return model

def handle_model(train_loader, val_loader, settings):
    model = initialize_model(settings)

    model_exists = check_model_exists(settings["neural_net"]["model_type"])

    if model_exists and settings["execution"]["cached_model"]:
        model = load_pretrained_model(model, settings["neural_net"]["model_type"])
    else:
        if not model_exists:
            print("No saved models present, running training...")
            if not os.path.exists("cached_models"):
                os.makedirs("cached_models")
        model = train_and_save_model(model, train_loader, val_loader, settings)

    return model
