# bash ./infra/scripts/chunk-audio.sh
SOURCEROOT="$HOME/Audio_Bank/Sources"
folders=$(find $SOURCEROOT -maxdepth 1 -mindepth 1 -type d)
count=0
for d in $folders; do
  cd $d
  echo $d
  echo $count
  count=$((count+1))
  sox $d/*.wav $d/concat.wav
  if [ $? -eq 0 ]; then
    parentname="$(basename "$(dirname "$d/concat.wav")")"
    length=$(soxi -D $d/concat.wav)
    length=${length/\.*}
    length=$((length / 5 * 5))
    sox $d/concat.wav $d/output.wav trim 0 $length
    find . -type f ! -name 'output.wav' -delete
    sox output.wav "${parentname}-%1n.wav" trim 0 5 : newfile : restart
    rm output.wav
  else
    rm $d/concat.wav
    echo $d >> ~/error_log.txt
  fi
done
