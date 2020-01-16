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

import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy.signal import bilinear
from scipy.signal import butter
from scipy.signal import decimate
from scipy.signal import hilbert
import numpy as np
from numpy import pi, polymul
from pydub import AudioSegment
import matplotlib.pyplot as plt

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

# Load the audio file
audio = AudioSegment.from_file(sys.argv[1])

# Uncomment below if you wish to also speed up the playback or see pydub docs for more effects
#audio = AudioSegment.speedup(audio, playback_speed=2)

# Get the sample rate and numpy array of the sound data
fs = audio.frame_rate
types = [np.uint8, np.int16, np.int32, np.int32]
data = np.fromstring(audio._data, types[audio.sample_width - 1])
temp = []
for ch in list(range(audio.channels)):
    temp.append(data[ch::audio.channels])
x = np.array(temp).T
x = x.flatten()

# Plot 1 - unmodified original audio
plt.subplot(3, 1, 1)
plot_signal(x, fs)
plt.title("Original audio")

# Apply A-weighting first
b, a = A_weight(fs)
y = filtfilt(b, a, x)

# Plot 2 - A-weighting applied to samples
plt.subplot(3, 1, 2)
plot_signal(y, fs)
plt.title("A-weighted")

# Get an envelope
analytic_signal = hilbert(y)
y_env = np.abs(analytic_signal)

# Plot 3 - envelope
plt.subplot(3, 1, 3)
plot_signal(y_env, fs)
plt.title("Envelope")
# Uncomment below if you wish to see the plots
#plt.show()
plt.savefig("{0:s}_processed.png".format(sys.argv[1][0:-4]))
plt.close()

window_length = 100
# TODO: Need a way to detect a good value for threshold (will depend on the audio and sample width)
threshold = int(sys.argv[2]) if len(sys.argv) == 3 else 10000000
segments = []

# Get non-silent segments
for i in range(0, len(y_env), window_length):
    Y = y_env[i:i+window_length+1]

    mean = Y.mean()*(1 + int((i-window_length) in segments)*0.5)
    if mean > threshold:
        segments.append(i)

# Plot for showing regions detected that have audio above threshold value
fig, ax = plt.subplots()
fig.set_size_inches(15, 6, forward=True)
ax.plot(np.linspace(0, len(x)/fs, len(x)), x)

start_seg = segments[0]
is_start_seg = True

# Plot regions of audio above threshold (There is certainly a more elegant way to do this.)
for i in range(0, len(segments)):
    if (i < len(segments)-1):
        # marks the end of a segment
        if is_start_seg and (segments[i+1]/fs-segments[i]/fs) > 0.13:
            plt.axvspan(start_seg/fs, segments[i]/fs, facecolor='g', alpha=0.5)
            is_start_seg = False
        # marks the start of a segment
        elif not is_start_seg and (segments[i+1]/fs-segments[i]/fs) <= 0.1:
            start_seg = segments[i]
            is_start_seg = True
    else:
        if is_start_seg:
            plt.axvspan(start_seg/fs, segments[i]/fs, facecolor='g', alpha=0.5)
            is_start_seg = False

plt.title("Detected silence")
# Uncomment below to show the plot
#plt.show()
plt.savefig("{0:s}_segments.png".format(sys.argv[1][0:-4]))
plt.close()

# Splice data and write out wav file
strOut = "{0:s}_cut.wav".format(sys.argv[1][0:-4])
out = np.array([], dtype=types[audio.sample_width - 1])

for i in segments:
    out = np.append(out, x[i:i+window_length])

# Final plot showing truncated output
plot_signal(out, fs)
plt.title("Truncated audio")
plt.savefig("{0:s}_trunc.png".format(sys.argv[1][0:-4]))

audio._data = out
audio.export(strOut, format='wav')
