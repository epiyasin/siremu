from dataset import MintDataset
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import FunctionTransformer
from settings import Settings

settings = Settings("config.json")

def prepare_nn_mint_data(settings):
    file_path = settings.mint_data_path
    emulator_data = MintDataset(input_file=file_path)

    test_pct = 0.2
    val_pct = 0.2

    train_pct = 1 - test_pct - val_pct
    train_size = int(train_pct * len(emulator_data))
    validation_size = int(val_pct * len(emulator_data))
    test_size = int(test_pct * len(emulator_data))

    train_dataset, validation_dataset, test_dataset = random_split(emulator_data,
                                                                   [train_size, validation_size, test_size])

    batch_size = 64
    num_workers = 16
    shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # MINT data are already scaled, this scaler just returns identity transformation
    scaler = FunctionTransformer()
    return train_loader, validation_loader, test_loader, scaler