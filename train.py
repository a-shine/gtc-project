from torch import nn, optim
from torch.utils.data import DataLoader
import torch


epochs = 5
criterion = nn.MSELoss()



def train(model, train_x, train_y, val_x, val_y, batch_size=32, device='cpu'):
    """
    Train the model with the given training and validation data.

    Args:
        model: The PyTorch model to train.
        train_x: The training data features.
        train_y: The training data labels.
        val_x: The validation data features.
        val_y: The validation data labels.
        batch_size: The batch size for training. Default is 32.
        device: The device to train on. Default is 'cpu'.
    """

    # Convert data to torch tensors and move to device
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    val_x = torch.from_numpy(val_x).float().to(device)
    val_y = torch.from_numpy(val_y).float().to(device)

    # Create data loaders
    train_data = DataLoader(list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(list(zip(val_x, val_y)), batch_size=batch_size)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set model to train mode
    model.train()

    # Loop over the epochs
    for epoch in range(epochs):
        # Loop over the training data in batches
        for batch_x, batch_y in train_data:
            # Zero the gradients
            optimizer.zero_grad()

            # Perform a forward pass
            outputs = model(batch_x)

            # Compute the loss
            loss = criterion(outputs, batch_y)

            # Perform a backward pass and update the model parameters
            loss.backward()
            optimizer.step()

        # Print the training loss for this epoch
        print(f"Epoch: {epoch + 1}, Training Loss: {loss.item()}")

        # Set model to evaluation mode
        model.eval()

        # Compute the validation loss
        with torch.no_grad():
            val_losses = []
            for val_batch_x, val_batch_y in val_data:
                val_outputs = model(val_batch_x)
                val_loss = criterion(val_outputs, val_batch_y)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

        # Print the validation loss for this epoch
        print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}")

        # Set model back to train mode
        model.train()
