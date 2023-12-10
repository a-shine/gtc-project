# flake8: noqa
import os

# Set backend env to torch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn
import torch.optim as optim
from keras_core import layers
import keras_core
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.load import load_data
from utils.params import set_param

# Model / data parameters
lookback = 10
forecast_horizon = 5
input_shape = (28, 28, 1)
learning_rate = 0.01
batch_size = 128
num_epochs = 1

exp = 'b726'
# exp = 'b698'
# exp = 'i417'
# exp = 'p4679'
# exp = 'p4581'
# exp = 'cascadia'
# exp = 'sim_b726'
# exp = 'sim_b698'
# exp = 'sim_i417'


def get_data() -> TensorDataset:
    dirs = {'main': os.getcwd()}
    params = set_param(exp)
    dirs['data'] = dirs['main'] + '/' + params['dir_data']
    x_raw, _, t, _, _ = load_data(exp, dirs, params)

    # Normalize data
    # TODO: Check effectiveness of this
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw.reshape(-1, 1))

    # As we are doing time series, y is just the next time step(s)
    # Define the lookback period and prepare input-output pairs

    # multistep prediction with multi-output model
    # if forecast_horizon > 1: # multi-step prediction else single-step prediction
    x = np.array([x_scaled[i:i + lookback] for i in range(len(x_scaled) - lookback - forecast_horizon + 1)])
    y = np.array([x_scaled[i + lookback:i + lookback + forecast_horizon] for i in
                  range(len(x_scaled) - lookback - forecast_horizon + 1)])

    # WARNING: Make sure shuffle is False for time series data
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)

    # Make sure time series x and y have a (train samples, lookback, features) shape
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    return dataset


def get_model():
    # Create the Keras model
    return MyModel().model


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = keras_core.Sequential([
            layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            # reshape to batch_size, forcast_horizon, 1)
            layers.Dense(1)


        ])

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            print(outputs.shape)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print loss statistics
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / 10}")
                running_loss = 0.0


def setup(current_gpu_index, num_gpu):
    # Device setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device("cuda:{}".format(current_gpu_index))
    dist.init_process_group(backend="nccl", init_method="env://", world_size=num_gpu, rank=current_gpu_index, )
    torch.cuda.set_device(device)


def prepare(dataset, current_gpu_index, num_gpu, batch_size):
    sampler = DistributedSampler(dataset, num_replicas=num_gpu, rank=current_gpu_index, shuffle=False, )

    # Create a DataLoader
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False, )

    return train_loader


def cleanup():
    # Cleanup
    dist.destroy_process_group()


def main(current_gpu_index, num_gpu):
    # setup the process groups
    setup(current_gpu_index, num_gpu)

    #################################################################
    ######## Writing a torch training loop for a Keras model ########
    #################################################################

    dataset = get_data()
    model = get_model()

    # prepare the dataloader
    dataloader = prepare(dataset, current_gpu_index, num_gpu, batch_size)

    # Instantiate the torch optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate the torch loss function
    loss_fn = nn.MSELoss()

    # Put model on device
    model = model.to(current_gpu_index)
    ddp_model = DDP(model, device_ids=[current_gpu_index], output_device=current_gpu_index)

    train(ddp_model, dataloader, num_epochs, optimizer, loss_fn)

    ################################################################
    ######## Using a Keras model or layer in a torch Module ########
    ################################################################

    torch_module = MyModel().to(current_gpu_index)
    ddp_torch_module = DDP(torch_module, device_ids=[current_gpu_index], output_device=current_gpu_index, )

    # Instantiate the torch optimizer
    optimizer = optim.Adam(torch_module.parameters(), lr=learning_rate)

    # Instantiate the torch loss function
    loss_fn = nn.CrossEntropyLoss()

    train(ddp_torch_module, dataloader, num_epochs, optimizer, loss_fn)

    cleanup()


if __name__ == "__main__":
    # GPU parameters
    num_gpu = torch.cuda.device_count()

    print(f"Running on {num_gpu} GPUs")

    torch.multiprocessing.spawn(
        main,
        args=(num_gpu,),
        nprocs=num_gpu,
        join=True,
    )
