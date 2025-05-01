# Import Libraries
from random import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def load_data(
        # Define directory locations
        train_directory=r"C:\Users\AaronBurton\Projects\FinalYearProject\Prototype\images\train",
        validate_directory=r"C:\Users\AaronBurton\Projects\FinalYearProject\Prototype\images\validation",

        # Define the variables for the function
        img_size=48,
        batch_size=64,
        split=0.8,
        augment = True,
        workers = 4
):
    # Define Augmentations
    if augment:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # Perform the same augmentation to the validated photos
    validation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ImageFolder automatically assigns class indices based on subfolder names
    train_dataset = datasets.ImageFolder(root=train_directory, transform=transform)
    validation_dataset = datasets.ImageFolder(root=validate_directory, transform=validation)

    # Split training dataset into training and validation
    dataset_size = len(train_dataset)
    train_size = int(split * dataset_size)
    validation_size = dataset_size - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, validation_size])

    # Assemble the dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Storing class names (emotions)
    class_names = train_dataset.classes

    return train_loader, valid_loader, test_loader, class_names

# Splitting the dataset into training and testing sets
def split_data(x, y, test_size=0.2):
    return train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)

