# bash ./infra/scripts/init-dependencies.sh
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install bc
sudo apt-get install sox
sudo apt-get install tar
sudo apt-get install gcc
sudo apt-get install unzip
sudo apt-get install ffmpeg
sudo apt-get install python3-pip
sudo add-apt-repository ppa:jonathonf/ffmpeg-3
sudo apt update && sudo apt upgrade
conda install git
pip install git+https://github.com/python-acoustics/python-acoustics.git
pip install git+https://github.com/wiseman/py-webrtcvad.git
conda install -c anaconda cython
conda install -c anaconda numpy
conda install -c anaconda scipy
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
conda install -c conda-forge librosa
conda install pytorch torchvision -c pytorch
conda install -c auto pydub
LIBROOT="$HOME/iMic/src/lib"
cd "$LIBROOT" && mkdir lib && git clone https://github.com/KernelLabs/py-webrtcvad.git
SCRIPTROOT="$HOME/iMic/infra/scripts"
cd "$SCRIPTROOT" && chmod u+x *.sh
