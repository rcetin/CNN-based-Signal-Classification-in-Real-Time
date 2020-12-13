#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:40:58 2019

@author: rcetin
"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import glob
import subprocess
import os
import time
import datetime
import gc
import time

# Number of samplepoints
np.set_printoptions(threshold=np.inf)

homepath = "/home/root"
spect_save_path = "{}/e310_test_images".format(homepath)
spect_save_prefix = "testdata_fft512_cf2440_"
general_counter = 0

test_data_path = "{}/test_data".format(homepath) # file to be splitted

splitted_prefix = "splitted_rf_data_"
splitted_files_save_path = "{}/splitted".format(homepath)
subprocess.call(["mkdir", "-p", splitted_files_save_path])

window_size = 512

split_size = "2M"

def split_files(file_path, split_prefix):
    files = glob.glob("{}".format(file_path))

    for current_file in files:
        huge_file = "{}".format(current_file)
        dest_dir = "{}/{}".format(splitted_files_save_path, split_prefix)
        #split huge data and remove last piece because its 1M!
        subprocess.call(["split", "-b", split_size, huge_file, dest_dir])
        splitted_files = glob.glob("{}/*".format(splitted_files_save_path))
        splitted_files = sorted(splitted_files)

        os.remove(splitted_files[-1])   # remove last piece
        del splitted_files[-1]  # delete it from array
    


def get_spec(name, counter):
    start = time.time()  
    
    file_name = name
    print("Reading sample file: {}".format(file_name))

    samples = np.fromfile(file_name, dtype=np.complex64) # Read samples as complex64
    samples = np.abs(samples)
    samples = (20 * np.log10(samples)).clip(-120)

    Y = []
    for i in range(int(len(samples) / window_size)):
        aa = samples[i * window_size: (i+1)*window_size]
        Y.append(aa)
    Y = np.column_stack(Y)
    
    plt.figure(figsize=(0.128, 0.128), dpi=100) 
    print("Sample count: {}".format(Y.shape[1]))
    plt.pcolormesh(Y, vmin=-120, vmax=0)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig_name = "{0}/{1}{2}_{3}.png".format(spect_save_path, spect_save_prefix, datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S'),counter)
    print("Generating spectrogram figure: {}".format(fig_name))
    plt.savefig(fig_name, box_inches='extent',pad_inches=0, dpi=1000)
    plt.close()
    plt.clf()
    gc.collect()
    stop = time.time()

    print("Duration: {} seconds\n-------------------------------------------------\n".format(stop - start))

split_files(test_data_path, splitted_prefix)
file_names = glob.glob("{}/*".format(splitted_files_save_path))
file_names =sorted(file_names)

for n, c in zip(file_names ,np.arange(len(file_names))):
    get_spec(name=n,counter=c)   
