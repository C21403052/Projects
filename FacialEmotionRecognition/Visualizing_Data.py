import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def create_confusion_matrix(model, loader, device, classes):
    # Set the model to evaluate mode
    model.eval()
    # Make to empty lists for the predictions and labels
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Build confusion matrix
    cfmtx = confusion_matrix(all_labels, all_predictions)
    cfmtx_df = pd.DataFrame(cfmtx, index=classes, columns=classes)

    # Plot confusion matrix
    plt.figure(figsize = (15,11))
    sns.set_theme(font_scale=1.5)
    sns.heatmap(cfmtx_df, annot=True, cmap='Reds')
    plt.title("Confusion Matrix", fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.show()

    return cfmtx_df

