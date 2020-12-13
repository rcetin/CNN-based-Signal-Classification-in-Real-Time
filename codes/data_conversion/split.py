#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import glob
import subprocess
import os
import gc
file_names = glob.glob("/tmp/test_data/*")
file_names =sorted(file_names)
print(file_names)

split_size = "2M"

for current_file in file_names:
    huge_file = "{}".format(current_file)
    dest_dir = "{}_".format(current_file)
    #split huge data and remove last piece because its 1M!
    subprocess.call(["split", "-b", split_size, huge_file, dest_dir])
    splitted_files = glob.glob("{}_*".format(current_file))
    splitted_files = sorted(splitted_files)

    for i in splitted_files:
        print("File: {}".format(i))

    if os.path.getsize(splitted_files[-1]) < 2000000:
        os.remove(splitted_files[-1])   # remove last piece
        del splitted_files[-1]  # delete it from array