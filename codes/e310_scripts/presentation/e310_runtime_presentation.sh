#!/bin/sh
# USRP dataset collection script
# rcetin

# E310 Runtime unit test for CNN

VERSION="1.0.0"
HOME="/home/root/"
SPLITTED_PATH="/home/root/splitted"
COL_CODE_PATH=$HOME"rfnoc_periodogram_runtime.py"
COL_CODE_OUT_DATA=$HOME"test_data"
SPECT_GENERATE_SCRIPT=$HOME"periodogram_to_targetdata_e310.py"
CNN_PREDICT_SCRIPT=$HOME"e310_cnn_prediction_v2.py"

echo -e "${YELLOW}\nUSRP Wi-fi data collection and CNN Prediction Script"
echo -e "Version: $VERSION"
echo -e "by rcetin${NOCOLOR}\n\n"

trap ctrl_c INT

function ctrl_c() 
{
  echo "Exiting application..."
  exit
}


source "/home/root/localinstall/setup_env.sh"
python $COL_CODE_PATH	# run collection script
python $SPECT_GENERATE_SCRIPT # Generate spectrograms
python $CNN_PREDICT_SCRIPT
