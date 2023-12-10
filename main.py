# import datetime
# import os
#
# import numpy as np
#
# from loading_data import train_test_split_forecasting, get_data
# from lstm_model import MyLSTM
# from train import train
# from dsp import top_n_frequencies, to_frequency_domain, butter_lowpass_filter
# import matplotlib.pyplot as plt
# import torch
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # exp = 'b726'
# exp = 'b698'
# # exp = 'i417'
# # exp = 'p4679'
# # exp = 'p4581'
# # exp = 'cascadia'
# # exp = 'sim_b726'
# # exp = 'sim_b698'
# # exp = 'sim_i417'
#
#
#
# if __name__ == "__main__":
#     print(f"Using device: {device}")
#
#     # Load data
#     X, t = get_data(exp, visualise=False)
#     train_x, train_y, test_x, test_y = train_test_split_forecasting(X, t, input_sequence_length=1000, forecast_horizon=500, visualise=False)
#
#     print(X.shape)  # shape of raw data
#     print(train_x.shape)  # shape of train data
#     print(train_y.shape)  # shape of train labels
#
#     # Signal analysis (DSP)
#     # a, f = to_frequency_domain(X, t, visualise=False)
#     # top_n_frequencies(a, f, 10)  # top 10 frequencies and their amplitudes
#     # # low pass filter to remove noise
#     # # Apply the filter to the signal in your main function
#     # # FIXME: This is not working
#     # # X_filtered = butter_lowpass_filter(X, 30, freq, 6)
#     # # top_n_frequencies(*to_frequency_domain(X_filtered, t), 10)
#
#     # Load architecture
#     model = MyLSTM(input_size=1, hidden_size=100, output_size=1, num_layers=1).to(device)
#
#     # Train model
#     # train(model, train_x, train_y, val_x, val_y, batch_size=32, device=device)
#
#     # Save model
#     # os.makedirs('models', exist_ok=True)
#     # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#     # torch.save(model.state_dict(), f'models/{exp}_{timestamp}.pt')
#
#     # test model (only needed if train, val, test split)
#     # test performance on test set
#     # test(model, test_x, test_y, device=device)
#
#     # Run inference
#     # load model from file
#     model.load_state_dict(torch.load('models/b698_20231205191533.pt'))
#     # Take a random sample from the test set
#     idx = np.random.randint(len(test_x))
#     sample_x = test_x[idx]
#     sample_y = test_y[idx]
#     # FIXME: What should the window lenght and forecast horizon be? At the moment they are different and so I have to truncate the output
#     # the problem is that the training tensor is of shape (batch_size, sequence_length, input_size) and the output tensor is of shape (batch_size, sequence_length, output_size)
#     # and the inference tensor is of shape (sequence_length, input_size) so I have to truncate the output
#     y_pred = model(torch.from_numpy(sample_x).float().to(device)).cpu().detach().numpy()
#     plt.plot(sample_y, label='True')
#     plt.plot(y_pred, label='Predicted')
#     plt.legend()
#     plt.show()
