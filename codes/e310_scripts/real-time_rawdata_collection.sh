#!/bin/sh
# USRP dataset collection script
# rcetin

VERSION="1.0.0"
HOME="/home/root/"
COL_CODE_PATH=$HOME"rfnoc_periodogram.py" 
COL_CODE_OUT_DATA=$HOME"test_data"
DATA_NAME="80211b_test_"
#DATA_NEW_NAME_PREFIX=$HOME"/wifi_data_"
DATA_COUNT=10

HOST_DEST="/home/rcetin/newdata_060819/afteriir/test/80211b_test_dataset/"
HOSTNAME="rcetin"
HOST_IP="192.168.10.1"

data_counter=0
control=0
currentFreq=2457e6

RED="\033[1;31m"
YELLOW="\033[1;33m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

echo -e "${YELLOW}\n\nUSRP Wi-fi data collection script"
echo -e "Version: $VERSION"
echo -e "by rcetin${NOCOLOR}\n\n"

trap ctrl_c INT

function ctrl_c() 
{
  echo "Exiting application..."
  exit
}

source "/home/root/localinstall/setup_env.sh"

counter=0
while [[ $counter -lt $DATA_COUNT ]]; do
	echo -e "Data collection starts at freq: $currentFreq\n"
	python $COL_CODE_PATH
	DATE=`date '+%Y%m%d_%H%M%S'`
	filepathname=$HOME$DATA_NAME$DATE

	filename=$DATA_NAME$DATE
	mv $COL_CODE_OUT_DATA $filepathname
	echo -e "Data collection is completed. Counter: {$counter} . Data is sending to host. \nFile: $filepathname"
	echo -e "Host destination: $HOST_DEST$filename\n"
	scp $filepathname $HOSTNAME@$HOST_IP:$HOST_DEST$filename
	rm $filepathname	# delete the dataset
#	echo -e "\n\n${GREEN}New data collection will start in 2 seconds ..."
#	echo -e "You can exit application via 'ctrl + C' in 2 seconds${NOCOLOR}"
	echo -e "----------------------------------------------------\n"
	counter=$((counter + 1))
#	sleep 2
done


