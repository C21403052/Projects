import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
import CNN_Model as cnn
import Process_Dataset as pds
import numpy as np
from sklearn.utils import compute_class_weight
from skopt import gp_minimize
from skopt.space import Real, Integer

class HyperparameterTuner:
    # Define the objects variables
    def __init__(self, trials=30):
        self.trials = trials
        self.best_params = None

    def objective_function(self, params):
        # Separate the parameters
        initial_lr, weight_decay, dropout_rate, batch_size = params

        # Load datasets
        training_set, validation_set, _, class_names = pds.load_data(
            augment=True, balance=True, batch_size=int(batch_size)
        )

        num_classes = len(class_names)

        # Compute the weight of each class (check for imbalance)
        labs = []
        for batch in training_set:
            _, labels = batch  # Extract labels from batch
            labs.extend(labels.numpy())  # Convert tensors to NumPy array and extend list

        unique_classes = np.unique(labs)  # Convert labels to numpy array and get unique class labels
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labs)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Check the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = class_weights.to(device)

        # Generate the model
        model = cnn.create_CNN_model(num_classes).to(device)

        # Define the data loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

        # Introduce Mixed Precision Training
        scaler = amp.GradScaler(enabled=(device.type =='cuda'))

        best_valid_loss = float("inf")
        for epoch in range(10):
            model.train()
            running_loss = 0.0
            for inputs, labels in training_set:
                inputs, labels = inputs.to(device), labels.to(device)

                with amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                running_loss += loss.item()

            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_set:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    valid_loss += loss.item()

            valid_loss /= len(validation_set)
            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), "BackUp Files/Best_FER_Model.pth")

        return best_valid_loss  # Minimize validation loss

    def tune_hyperparameter(self):
        # Define Hyperparameter Search Space
        search_space = [
            Real(1e-5, 1e-2, "log-uniform"),
            Real(1e-6, 1e-3, "log-uniform"),
            Real(0.2, 0.5),
            Integer(32, 128)
        ]

        # Perform Bayesian Optimization using Gaussian Process
        result = gp_minimize(self.objective_function, search_space, n_calls=self.trials, random_state=42)

        # Extract the Best Parameters
        self.best_params = result.x
        print(f"Best Hyperparameters:\n"
              f"Learning rate: {self.best_params[0]:.6f}\n"
              f"Weight decay: {self.best_params[1]:.6f}\n"
              f"Dropout rate: {self.best_params[2]:.6f}\n"
              f"Batch Size: {int(self.best_params[3])}\n")


        return self.best_params

