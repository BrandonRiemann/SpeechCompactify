"""
compactify.py

Shortens the length of an audio file by removing silence and increasing the speed.

Note: Uses pydub library which loads into memory (only small files have been tested)

Other methods certainly can improve upon this naive attempt, especially since I have little
experience in audio signal processing. However, it works for my intended purposes. You might need
to play with the threshold value or window length.

The current method is to apply A-weighting to the signal to help identify silence better.
We then use the Hilbert transform on the filtered samples to get an envelope which we compare
against a threshold value. More information here:
https://www.mathworks.com/help/dsp/examples/envelope-detection.html

Brandon Sachtleben

TODO:
* Handle more diverse cases such as background noise or multiple sources of noise.
* Improve performance.
* Rewrite without pydub if possible (I had some issues with reading using scipy.wavfile)
"""

import os
import sys

if len(sys.argv) not in [2, 3]:
    print("Usage: python compactify.py [audio filename] [threshold value (optional)]")
    sys.exit(1)

if not os.path.isfile(sys.argv[1]):
    raise Exception("Cannot find file")

from scipy.signal import filtfilt
from scipy.signal import bilinear
from scipy.signal import hilbert
import numpy as np
from numpy import pi, polymul
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

def A_weight(fs):
    """
    Coefficients and formula based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4331191/
    """
    o = 2*pi*np.array([20.598997, 107.65265, 737.86223, 12194.217])
    G = -2.0

    num = [G*o[3]**2, 0, 0, 0, 0]
    denom = polymul(polymul(polymul([1, o[0]], [1, o[0]]), polymul([1, o[3]], [1, o[3]])),
                    polymul([1, o[1]], [1, o[2]]))

    return bilinear(num, denom, fs)

def plot_signal(signal, fs):
    plt.plot(np.linspace(0, len(signal)/fs, len(signal)), signal)

class Audio:
    def __init__(self, filename):
        # Load the audio file
        self.audio = AudioSegment.from_file(filename)

        # Get the sample rate and numpy array of the sound data
        self.fs = self.audio.frame_rate
        self.types = [np.uint8, np.int16, np.int32, np.int32]
        x = np.fromstring(self.audio._data, self.types[self.audio.sample_width - 1])
        temp = []
        for ch in list(range(self.audio.channels)):
            temp.append(x[ch::self.audio.channels])
        self.data = np.array(temp).T
        self.data = self.data.flatten()

        # Parameters
        self.window_length = 100
        self.threshold = int(sys.argv[2]) if len(sys.argv) == 3 else 10000000

    def remove_silence(self, plot = False):
        # Progress bar
        pbar_step = len(self.data)
        pbar_total = 5*pbar_step
        pbar = tqdm(total = pbar_total)

        # Plot 1 - unmodified original audio
        plt.subplot(3, 1, 1)
        plot_signal(self.data, self.fs)
        plt.title("Original audio")

        # Apply A-weighting first
        b, a = A_weight(self.fs)
        y = filtfilt(b, a, self.data)

        pbar.update(pbar_step)

        # Plot 2 - A-weighting applied to samples
        plt.subplot(3, 1, 2)
        plot_signal(y, self.fs)
        plt.title("A-weighted")

        # Get an envelope
        analytic_signal = hilbert(y)
        y_env = np.abs(analytic_signal)

        pbar.update(pbar_step)

        # Plot 3 - envelope
        plt.subplot(3, 1, 3)
        plot_signal(y_env, self.fs)
        plt.title("Envelope")

        if plot:
            plt.show()

        plt.savefig("{0:s}_processed.png".format(sys.argv[1][0:-4]))
        plt.close()

        segments = []

        # Get non-silent segments
        for i in range(0, len(y_env), self.window_length):
            Y = y_env[i:i+self.window_length+1]

            mean = Y.mean()*(1 + int((i-self.window_length) in segments)*0.5)
            if mean > self.threshold:
                segments.append(i)

            pbar.update(self.window_length * int(i > 0))

        pbar.update(len(y_env) - self.window_length * np.floor(len(y_env)/self.window_length))

        # Plot for showing regions detected that have audio above threshold value
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 6, forward=True)
        ax.plot(np.linspace(0, len(self.data)/self.fs, len(self.data)), self.data)

        start_seg = segments[0]
        is_start_seg = True

        # Plot regions of audio above threshold (There is certainly a more elegant way to do this.)
        for i in range(0, len(segments)):
            if (i < len(segments)-1):
                # marks the end of a segment
                if is_start_seg and (segments[i+1]/self.fs-segments[i]/self.fs) > 0.13:
                    plt.axvspan(start_seg/self.fs, segments[i]/self.fs, facecolor='g', alpha=0.5)
                    is_start_seg = False
                # marks the start of a segment
                elif not is_start_seg and (segments[i+1]/self.fs-segments[i]/self.fs) <= 0.1:
                    start_seg = segments[i]
                    is_start_seg = True
            else:
                if is_start_seg:
                    plt.axvspan(start_seg/self.fs, segments[i]/self.fs, facecolor='g', alpha=0.5)
                    is_start_seg = False

        pbar.update(pbar_step)

        plt.title("Detected silence")

        if plot:
            plt.show()

        plt.savefig("{0:s}_segments.png".format(sys.argv[1][0:-4]))
        plt.close()

        # Splice data segments
        out = np.array([], dtype=self.types[self.audio.sample_width - 1])

        for i in segments:
            out = np.append(out, self.data[i:i+self.window_length])

        # Final plot showing truncated output
        plot_signal(out, self.fs)
        plt.title("Truncated audio")
        plt.savefig("{0:s}_trunc.png".format(sys.argv[1][0:-4]))

        pbar.update(pbar_total - pbar.n)
        pbar.close()

        return out

    def export(self, filename, data):
        self.audio._data = data
        self.audio.export(filename, format='wav')

audio = Audio(sys.argv[1])
data = audio.remove_silence(plot = True)
strOut = "{0:s}_cut.wav".format(sys.argv[1][0:-4])
audio.export(strOut, data)
