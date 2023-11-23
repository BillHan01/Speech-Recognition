# created by Han Jiarui
# edited on Oct 15 2023
# This is code of a MFCC process for assignment 1
# of 2023 Fall ASR course at SSE, Tongji University

import numpy as np
import librosa      # used for reading voice files
from scipy.fftpack import fft, dct
import matplotlib.pyplot as plt
import pandas as pd

# read voice file 'mfcc.wav'
y, sr = librosa.load('mfcc.wav', sr=None)   # the sampling rate of this file is 48000


# 1. Pre-emphasis
def pre_emphasis(signal, coefficient=0.97):
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

y_preemphasized = pre_emphasis(y)

# draw raw audio signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.title("Original Audio Signal")
plt.xlabel("sample points")
plt.ylabel("amplitude")

# draw audio signal after pre-emphasis
plt.subplot(2, 1, 2)
plt.plot(y_preemphasized, color='r')
plt.title("Audio Signal after Pre-emphasis")
plt.xlabel("sample points")
plt.ylabel("amplitude")

plt.tight_layout()
plt.show()

print('1 - pre-emphasis completed！')


# 2. Perform framing and windowing
frame_size = 0.025  # seconds
frame_stride = 0.01  # seconds

# calculate the length and step size of the frame
frame_length = int(round(frame_size * sr))
frame_step = int(round(frame_stride * sr))
signal_length = len(y_preemphasized)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

# numpy array for frames
# Pad the signal with zeros
pad_signal_length = num_frames * frame_step + frame_length
padded_signal = np.pad(y_preemphasized, (0, pad_signal_length - signal_length), 'constant')
# Create frame indices for extraction
indices = np.tile(np.arange(frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = padded_signal[indices]

frames *= np.hamming(frame_length)

print("Frame size: {:.3f} seconds".format(frame_size))
print("Frame stride: {:.3f} seconds".format(frame_stride))
print("Frame length: {} samples".format(frame_length))
print("Frame step: {} samples".format(frame_step))
print("Total number of frames: {}".format(num_frames))

plt.figure(figsize=(10, 4))
plt.plot(np.hamming(frame_length))
plt.title("t-domain Waveform of Hamming Window")
plt.xlabel("sample points")
plt.ylabel("amplitude")
plt.show()

print('2 - windowing completed！')


# 3. Perform short time Fourier transform
n_FFT = 1300  # size of FFT
hop_length = frame_step  # time interval between frames

stft_matrix = np.array([np.fft.fft(frame, n=n_FFT) for frame in frames]) # initialize STFT matrix
stft_matrix = stft_matrix[:, :n_FFT // 2 + 1]    # we only need the positive portion

# calculate amplitude spectrum of STFT
magnitude = np.abs(stft_matrix)
# calculate energy spectrum
energy = magnitude ** 2

# Visualize the amplitude spectrum of STFT
plt.figure(figsize=(10, 6))
plt.imshow(np.log1p(magnitude.T), cmap='viridis', aspect='auto', origin='lower')
plt.title("STFT Magnitude Spectrum")
plt.xlabel("Frames")
plt.ylabel("Frequency Bins")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

print('3 - STFT completed!')


# 4. Applying Mel-filter bank
n_filter = 40      # number of filters

# dividing Mel scale
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filter + 2)
hz_points = (700 * (10**(mel_points / 2595) - 1))   # convert Mel back to Hz

# Calculate the Mel filter bank
bin = np.floor((n_FFT + 1) * hz_points / sr).astype(int)  # Calculate the bin index of each Mel filter in FFT

# Initialize the matrix of fbank
fbank_matrix = np.zeros((n_filter, int(np.floor(n_FFT / 2 + 1))))

# Calculate the Mel filter bank weights using vectorized operations
for filt in range(1, n_filter + 1):
    left_boundary = bin[filt - 1]  # Left boundary
    middle = bin[filt]  # Middle
    right_boundary = bin[filt + 1]  # Right boundary

    # Calculate the weight of bin for each frequency using vectorized operations
    w1 = np.arange(left_boundary, middle)
    w2 = np.arange(middle, right_boundary)

    fbank_matrix[filt - 1, w1] = (w1 - bin[filt - 1]) / (bin[filt] - bin[filt - 1])
    fbank_matrix[filt - 1, w2] = 1 - (w2 - bin[filt]) / (bin[filt + 1] - bin[filt])

# Weight the energy of audio signals on the Mel scale using matrix multiplication
filter_banks = np.dot(energy, fbank_matrix.T)

# Ensure numerical stability by replacing zeros with epsilon
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

# Print the Mel filter bank output matrix
print("Mel-filter bank output matrix:")
print(filter_banks)

# Plot the Mel-filter bank
plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect='auto')
plt.title('Mel-filter Bank')
plt.xlabel('frames')
plt.ylabel('filter banks')
plt.show()

print('4 - Mel-filter bank completed!')


# 5. Log
filter_banks = 20 * np.log10(filter_banks)  # dB
print('5 - log completed!')


# 6. Discrete cosine transform (DCT)
num_coefficients = 12   # hyperparameter: number of coefficients to retain
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_coefficients + 1)]

# show the result of DCT
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect='auto')
plt.title('MFCC')
plt.show()

print('6 - DCT completed!')


# 7. Dynamic feature extraction
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

# print MFCC Delta feature matrix
print("\nMFCC Delta matrix:")
print(mfcc_delta)

# print MFCC Delta2 feature matrix
print("\nMFCC Delta-Delta (Delta2) matrix:")
print(mfcc_delta2)

print('7 - dynamic feature extract completed!')


# 8. Feature transformation
# CMN
mean = np.mean(mfcc, axis=0)
mfcc -= mean

# CVN
std = np.std(mfcc, axis=0)
mfcc /= (std + 1e-8)  # to avoid dividing by 0

print('8 - feature transformation completed!')


# 9. Result output
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect='auto')
plt.title('MFCC')
plt.show()

plt.imshow(np.flipud(mfcc_delta.T), cmap=plt.cm.jet, aspect='auto')
plt.title('MFCC Delta')
plt.show()

plt.imshow(np.flipud(mfcc_delta2.T), cmap=plt.cm.jet, aspect='auto')
plt.title('MFCC Delta2')
plt.show()

# print MFCC matrix
np.set_printoptions(precision=5, suppress=True, linewidth=150)
print("\nMFCC matrix:")
print(mfcc)

print('9 - outcome printing done！')

# export data to xlsx files
df1 = pd.DataFrame(mfcc)
df1.to_excel("final_mfcc.xlsx", index=False)
df2 = pd.DataFrame(mfcc_delta)
df2.to_excel("mfcc_delta.xlsx", index=False)
df3 = pd.DataFrame(mfcc_delta2)
df3.to_excel("mfcc_delta2.xlsx", index=False)