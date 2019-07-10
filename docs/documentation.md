# Audio_Net Documentation
First run iMic/infra/scripts/init-dependencies.sh. Then run iMic/infra/scripts/init-audiobank.sh to setup
Audio_Bank folder with normalized audio. See iMic/src/audio_preprocessing/directory_structure.txt for
an overview of the directory structure. Then run iMic/src/audio_preprocessing/prepreocess_audio.py to move
audio files from Audio_Bank/Raw to Audio_Bank/Sources (clips are chunked to 5 seconds). Then run iMic/src/audio_preprocessing/partition_data.py to create source file and mix CSVs in Audio_Bank/Meta. Run iMic/src/audio_net/model.py to train/validate model and run on test dataset. Actual sources/predicted sources/mix will be written as wav files to Audio_Bank/Output. Training/Validation
curves will be saved to iMic/src/audio_net/training_curves. iMic/src/audio_utils
has visualization tools for creating duration histograms and plots of spectrograms.
Important Papers: https://arxiv.org/pdf/1708.09588.pdf and https://arxiv.org/pdf/1703.06284.pdf
