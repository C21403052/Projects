# Import libraries
import torch.nn as nn
import torch.nn.functional as f

# Function to creat and return the CNN
def create_CNN_model(num_classes):
    # Class to define the CNN structure (i.e. Layers and pools)
    class EmotionCNN(nn.Module):
        # Initialize the layers
        def __init__(self, num_classes):
            # Call constructor from nn.Module
            super(EmotionCNN, self).__init__()

            """
            Define the Convolutional layers.
            Input is the number of channels of the input image.
            Output is the number of feature maps (filters) the layer will return
            Kernel size is 3x3
            Stride moves by 1 step for each kernel
            Padding just adds 1 pixel border around the input image
            """
            self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
            self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
            self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)

            # Batch normalization layers
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(64)
            self.bn5 = nn.BatchNorm2d(128)
            self.bn6 = nn.BatchNorm2d(128)
            self.bn_fc1 = nn.BatchNorm1d(256)

            # Pooling Layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 6 * 6, 256)
            self.fc2 = nn.Linear(256, num_classes)

            # Drops neurons to prevent overfitting
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout_fc1 = nn.Dropout(0.4)

        # Function to define the flow of inputted data through the network
        def forward(self, x):
            # Train and test loss accuracy and
            x = self.conv1(x)
            x = self.bn1(x)
            x = f.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = f.relu(x)
            x = self.pool(x)
            x = self.dropout1(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = f.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = f.relu(x)
            x = self.pool(x)
            x = self.dropout2(x)

            x = self.conv5(x)
            x = self.bn5(x)
            x = f.relu(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = f.relu(x)
            x = self.pool(x)
            x = self.dropout3(x)

            # The view function will reshape the tensor into an array for the fully connected layers
            x = x.view(x.size(0), -1)
            print("After Flattening:", x.shape)

            # Now apply the ReLU function the fully connected layer
            x = self.fc1(x)
            x = self.bn_fc1(x)
            x = f.relu(x)
            x = self.dropout_fc1(x)

            x = self.fc2(x)
            return x

    # Call the function to create a CNN model instance
    model = EmotionCNN(num_classes)
    # Pass the model
    return model
