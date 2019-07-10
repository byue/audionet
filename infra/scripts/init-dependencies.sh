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
sudo -H pip install webrtcvad
sudo pip install Cython
sudo pip install acoustics
sudo pip install numpy
sudo pip install scipy
sudo pip install pandas
sudo pip install matplotlib
sudo pip install librosa
sudo pip install torch torchvision
sudo pip install pydub
LIBROOT="$HOME/iMic/src/lib"
cd "$LIBROOT" && mkdir lib && git clone https://github.com/KernelLabs/py-webrtcvad.git
SCRIPTROOT="$HOME/iMic/infra/scripts"
cd "$SCRIPTROOT" && chmod u+x *.sh
