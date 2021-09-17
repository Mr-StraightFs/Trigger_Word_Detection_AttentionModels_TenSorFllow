import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython

# Listening to Some Data (Ausio Snippets)
IPython.display.Audio("./raw_data/activates/1.wav")
IPython.display.Audio("./raw_data/negatives/4.wav")
# Audio Graph
x = graph_spectrogram("audio_examples/example_train.wav")
# CHecking out the Data before and After the Spectrogram
_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio('./raw_data/')

# Load Video Segments Using PyDub
print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")


