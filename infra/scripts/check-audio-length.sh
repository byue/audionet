# bash ./infra/scripts/check-audio-length.sh
SOURCEROOT="$HOME/Audio_Bank/Sources"
folders=$(find $SOURCEROOT -maxdepth 1 -mindepth 1 -type d)
for d in $folders; do
  cd $d
  FILES_WAV=$(find . -name '*.wav')
  for f in $FILES_WAV; do
  	  length=$(soxi -D "$f")
  	  length=${length/\.*}
  	  if [ "$length" -ne "5" ]; then
  	  	echo "$f"
  	  fi
  done
done
