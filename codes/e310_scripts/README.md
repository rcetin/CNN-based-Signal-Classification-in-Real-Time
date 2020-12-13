# E310 SCRIPTS

*real-time_rawdata_collection.sh* script can be employed to collect real-time raw data from WLAN modem.

## Presentation

The *presentation* folder includes the all files that are necessary for full operation in the field. 

**rfnoc_periodogram_runtime.py:** Programs the FPGA with the synthesized image, which contains DPE modules with necessary computation engines. DPE operation starts by collecting raw IQ samples. Incoming samples are passed through *periodogram*, *SP-IIR* and *downsampling* blocks. Obtained samples are stored to a file with a pre-defined count.

**periodogram_to_targetdata_e310.py:** This file converts stored samples to dataset samples, which will be used as an input of CNN models. Converted dataset samples will be stored in corresponding files.

**e310_cnn_prediction_v2.py:** This file performs CNN classification operation. It reads the dataset sample and passes the sample through entire CNN layers to predict the class of given sample. It prints which class the sample belongs to the console.

**e310_runtime_presentation.py:** This file is a standalone file for run-time presentation. It calls above code files with a correct order to complete all steps for operation.