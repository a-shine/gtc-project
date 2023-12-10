import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter


def to_frequency_domain(X, t, visualise=False):
    # FFT of X to get the dominant frequency
    # Compute the FFT of the signal
    fft_result = np.fft.fft(X)

    # Generate the frequencies for the FFT
    freqs = np.fft.fftfreq(len(X), t[1] - t[0])

    # Only keep positive frequencies
    mask = freqs >= 0
    freqs = freqs[mask]
    fft_result = fft_result[mask]

    if visualise:
        # plot the FFT (amplitude frequency)
        plt.plot(freqs, np.abs(fft_result))
        plt.xlabel('Frequency [Hz]')

        plt.show()

    return freqs, fft_result


def top_n_frequencies(freqs, fft_result, n):
    # Compute the absolute values of the FFT results
    fft_abs = np.abs(fft_result)

    # Combine the frequencies and their corresponding FFT absolute values into a list of tuples
    freqs_and_fft = list(zip(freqs, fft_abs))

    # Sort this list in descending order based on the FFT absolute values
    freqs_and_fft.sort(key=lambda x: x[1], reverse=True)

    # Print the top 10 tuples from the sorted list
    print("Top 10 frequencies and their amplitudes:")
    for freq, amp in freqs_and_fft[:10]:
        print(f"Frequency: {freq} Hz, Amplitude: {amp}")

    return freqs_and_fft[:n]


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
