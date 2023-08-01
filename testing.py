import torch
from tqdm import tqdm

def run_emulator(model, test_loader):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Running emulator'):
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actual.append(targets.numpy())
            
    return predictions, actual
