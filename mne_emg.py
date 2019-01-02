import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join


import mne
import os.path as op
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
                          preload=True)
raw.set_eeg_reference('average', projection=True)  # set EEG average reference
raw.plot(block=True, lowpass=40)