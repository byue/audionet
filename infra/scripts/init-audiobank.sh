# bash ./infra/scripts/init-audiobank.sh
# Convert to 16000 Hz, Mono, wav
cd ~
mkdir Audio_Bank
cd Audio_Bank
mkdir Raw
mkdir Sources
mkdir Meta
mkdir Output
cd Raw

# Voxforge
wget https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz
tar -xvzf voxforge_corpus_v1.0.0.tar.gz && mv archive voxforge && rm voxforge_corpus_v1.0.0.tar.gz
cd voxforge
for filename in *.tgz; do
  tar -xvzf $filename;
  rm $filename;
done
cd ~/Audio_Bank/Raw

# Tatoeba
wget https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip
unzip tatoeba_audio_eng.zip && mv tatoeba_audio_eng tatoeba && rm tatoeba_audio_eng.zip

# TODO: add Ted-Lium steps for download/unzipping

# Convert mp3 to wav, 16000 Hz
cd ~/Audio_Bank/Raw
FILES_MP3=$(find . -name "*.mp3")
NUM_MP3=$(find . -name "*.mp3" | wc -l)
counter=0
for f in $FILES_MP3; do
  soxi "$f" &> /dev/null
  if [ $? -eq 0 ]; then
    OUTNAME="${f%.*}.wav"
    ffmpeg -i "$f" -ar 16000 -ac 1 "$OUTNAME" &> /dev/null
    counter=$((counter+1))
    n=$(($counter%1000))
    if [ $n -eq 0 ]; then
      echo "Converted mp3 $f to wav: $counter out of $NUM_MP3"
    fi
  else
    echo "corrupted mp3, removing $f"
  fi
  rm "$f"
done

FLAC=$(find . -name "*.flac")
NUM_FLAC=$(find . -name "*.flac" | wc -l)
counter=0
for f in $FLAC; do
  soxi "$f" &> /dev/null
  if [ $? -eq 0 ]; then
    OUTNAME="${f%.*}.wav"
    ffmpeg -i "$f" -ar 16000 -ac 1 "$OUTNAME" &> /dev/null
    counter=$((counter+1))
    n=$(($counter%1000))
    if [ $n -eq 0 ]; then
      echo "Converted flac $f to wav: $counter out of $NUM_FLAC"
    fi
  else
    echo "corrupted flac, removing $f"
  fi
  rm "$f"
done

# https://todd.1750studios.com/2017/06/28/ebu-r128-and-ffmpeg/
# http://k.ylo.ph/2016/04/04/loudnorm.html
FILES_WAV=$(find . -name "*.wav")
NUM_PROCESS=$(find . -name "*.wav" | wc -l)
counter=0
for f in $FILES_WAV; do
  soxi "$f" &> /dev/null
  if [ $? -eq 0 ]; then
    ffmpeg -guess_layout_max 0 -i "$f" -c:v copy -pass 1 -af loudnorm=I=-16:TP=-1.5:LRA=11 &> /dev/null
    ffmpeg -guess_layout_max 0 -i "$f" -c:v copy -pass 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 -y "$f" &> /dev/null
    if [ $? -eq 0 ]; then
      ffmpeg -i "$f" -ar 16000 -ac 1 "$f" &> /dev/null
      n=$(($counter%1000))
      if [ $n -eq 0 ]; then
        echo "Normalized: $counter out of $NUM_PROCESS"
      fi
    else
      echo "Error normalizing $f"
    fi
  else
    rm "$f"
  fi
done
