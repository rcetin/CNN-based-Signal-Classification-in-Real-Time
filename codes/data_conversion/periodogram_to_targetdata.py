#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import subprocess
import os
import gc
from PIL import Image

import threading
import matplotlib.cm as cm
# Number of samplepoints
np.set_printoptions(threshold=np.inf)
sp_type = "n_high"
spect_save_path = "/media/rcetin/42375AE8738FEA25/newdata_060819/afteriir/v3/80211{}_afteriir_v3_dataset/spects_gray/".format(sp_type)
spect_save_prefix = "{}_afteriir_gray_v2_sp_".format(sp_type)
general_counter = 0

subprocess.call(["mkdir", "-p", spect_save_path])

file_names = glob.glob("/media/rcetin/42375AE8738FEA25/newdata_060819/afteriir/v3/80211{}_afteriir_v3_dataset/splitted/*".format(sp_type))
file_names =sorted(file_names)
print(file_names)

window_size = 512

def get_spec(name, counter):

    file_name = name
    print("Reading sample file: {}".format(file_name))

    samples = np.fromfile(file_name, dtype=np.complex64) # Read samples as complex64
    samples = np.abs(samples)
    samples = (20 * np.log10(samples)).clip(-120)

    Y = []
    for i in range(int(len(samples) / window_size)):
        aa = samples[i * window_size: (i+1)*window_size]
        Y.append(aa)

    Y = np.rot90(Y, axes=(0,1)) # convert image counter clockwise
    
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
    Y = Y*255
    Y = Y.astype('uint8')
    img = Image.fromarray(Y)
    img = img.resize((128,128))
    fig_name = "{0}{1}{2}.png".format(spect_save_path, spect_save_prefix, counter)

    print("Generating spectrogram figure: {}".format(fig_name))

    img.save(fig_name)

    print("-------------------------------------------------\n")

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=5)(delayed(get_spec)(name=n, counter=c) for n, c in zip(file_names ,np.arange(len(file_names))))