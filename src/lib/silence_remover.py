import sys
import os
LIBROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(LIBROOT, 'py-webrtcvad/'))
import example as vad

# takes source wav file and replaces with chopped wav files with silence removed. returns number
# of final files with or without replacement
def remove_silences(aggressiveness, source_wav_path):
	"""
	Removes intervals of silences and generates several smaller wav files from a wav file.
	:AssertionError: 		wav file must have a sample rate of 8,000, 16,000, or 32,000, must be mono too.
	:param agressiveness:	int
							How sensitive filter is to wav files (0 to 3)
	:param wav_path:		string
							path to the wav file
	"""
	args = [aggressiveness, source_wav_path]
	return vad.main(args)
