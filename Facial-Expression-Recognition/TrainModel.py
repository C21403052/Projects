# Import libraries
import os

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight

import CNN_Model as cnn
import Process_Dataset as pds
from Visualizing_Data import create_confusion_matrix
from HyperparameterTuner import HyperparameterTuner

if __name__ == '__main__':

    # Run Hyperparameter Tuning
    #tuner = HyperparameterTuner(trials = 30)
    #best_hyperparameters = tuner.tune_hyperparameter()

    # Function to train the dataset
    # Epoch = number of passes
    def train(
            epochs=100,
            patience=10,
            accumulation_steps=2,
            initial_lr=0.0003,
            weight_decay=0.0001,
            label_smoothing=0.1
    ):
        # Load in the data
        train_loader, validate_loader, _, class_names = pds.load_data(
            augment=True,
            balance=True
        )

        # Model Directory
        model_dir = r"/FACIAL_EXPRESSION_RECOGNITION_APP\models"
        os.makedirs(model_dir, exist_ok=True)

        # Determine number of output classes
        num_classes = len(class_names)

        # Collect labels from trained loader for class weights
        labels = []
        for _, label in train_loader.dataset:
            labels.append(label)
        labels = np.array(labels)


        # Compute class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

        # Detect Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


        model = cnn.create_CNN_model(num_classes).to(device)

        # Cross compare the predictions with the labels
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # Set the loss function and optimizer
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

        # Learning rate scheduler (reduce on plateau of val loss)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

        # Mixed Precision Training
        scaler = amp.GradScaler(enabled=(device.type == 'cuda'))

        # Create variables to track accuracy and loss
        train_losses = []
        valid_losses = []
        train_acc = []
        valid_acc = []
        best_valid_loss = float("inf")
        patience_counter = 0

        # Loop to train for each epoch defined
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Reset optimizer before gradient accumulation
            optimizer.zero_grad()

            # Loop through the dataset in batches
            for step, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device) # Move data to the GPU if available

                with amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    # Get the predictions from the model
                    outputs = model(inputs)
                    # Calculate the loss and gets highest probability from labels
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()


                # Adjust for the loss of epoch
                running_loss += loss.item()

                # Calculate the training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # Ensure computations happen on CPU for accuracy calculations
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100.0 * correct / total
            train_acc.append(train_accuracy)

            # Validate the model
            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in validate_loader:
                    inputs, labels = inputs.to(device), labels.to(device) # Move data to GPU
                    with amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            valid_loss = valid_loss / len(validate_loader)
            valid_losses.append(valid_loss)
            valid_accuracy = 100 * correct / total
            valid_acc.append(valid_accuracy)
            scheduler.step(valid_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: "
                  f"{train_accuracy:.2f}%, Val Loss: {valid_loss:.4f}, Val Accuracy: {valid_accuracy:.2f}%")

            model_name = os.path.join(model_dir, f"Model{epoch+1}.pth")
            torch.save(model.state_dict(), model_name)
            print(f"Model saved at {model_name}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'BackUp Files/Best_FER_Model.pth')
                print(f"->New Best Model Trained!")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early Stopping")
                break

        # Plot Training and validation loss and accuracy
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(14, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Training Loss')
        plt.plot(epochs_range, valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_acc, label='Training Accuracy')
        plt.plot(epochs_range, valid_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

        # Save model to Best Model
        model.load_state_dict(torch.load('BackUp Files/Best_FER_Model.pth'))
        model.to(device)

        # Create confusion matrix
        cfmtx_df = create_confusion_matrix(
            model=model,
            loader=validate_loader,
            device=device,
            classes=class_names
        )

        print("\Confusion Matrix DataFrame:\n", cfmtx_df)

    # return model
    train()