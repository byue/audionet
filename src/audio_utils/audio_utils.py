# Directory/file management
import os
from os.path import basename, isdir
from os import walk
import sys
# Math
import numpy as np
# Audio Management
from scipy import signal
from scipy.io import wavfile
# Visualization
import matplotlib.pyplot as plt
import librosa.display
# Bash scripting
import subprocess
import re
import math
import argparse


BANK_ROOT = os.path.join(os.path.expanduser('~'), 'Audio_Bank')
SOURCE_ROOT = os.path.join(BANK_ROOT, 'Sources')
META_ROOT = os.path.join(BANK_ROOT, 'Meta')

# given a path to root, creates a histogram of all audio clip durations in seconds inside root and saves to save_dir.
def plot_durations_histogram(root, save_dir):
	min = sys.float_info.max
	max = -1
	num_files = sum([len(files) for r, d, files in os.walk(root)])
	counter = 1
	durations = []
	total_duration = 0.0
	for (dirpath, dirnames, filenames) in walk(root):
		for file in filenames:
			if file.endswith('.wav'):
				file_path = os.path.join(dirpath, file)
				process = subprocess.Popen(['ffmpeg',  '-i', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
				stdout, stderr = process.communicate()
				match = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL)
				if match is None:
					print(file + " is nontype")
					continue
				matches = match.groupdict()
				seconds = float(matches['seconds'])
				if seconds < 12.0:
					durations.append(seconds)
				total_duration += seconds
				if seconds > max:
					max = seconds
				if seconds < min:
					min = seconds
				if counter % 1000 == 0:
					mean_duration = total_duration / counter
					print("progress: " + str(counter) + "/" + str(num_files) + ", min: " + str(min) + ", max: " + str(max) + ", mean: " + str(mean_duration))
					n, bins, patches = plt.hist(durations, bins=range(0, math.ceil(max)))
					plt.xticks(np.arange(0, 12, 1.0))
					plt.xlabel('Seconds')
					plt.ylabel('Number of Samples')
					plt.title('Sample Durations')
					plt.savefig(os.path.join(save_dir, "sample_durations_histogram.png"))
				counter += 1
	n, bins, patches = plt.hist(durations, bins=range(0, math.ceil(max)))
	plt.xticks(np.arange(0, 12, 1.0))
	plt.xlabel('Seconds')
	plt.ylabel('Number of Samples')
	plt.title('Sample Durations')
	plt.savefig(os.path.join(save_dir, "sample_durations_histogram.png"))
	plt.show()


# file_name: input wav file path
# save_dir: output folder (current directory is ./ or .) to save .png grams. Does not save .png if only arg1 is present.
def plot_spectrograms(file_name, save_dir):
	# Note: To convert scipy wavfile time series to librosa time series, do samples = samples * -3.0517578e-05
	sample_rate, samples, time_length = readWavFloat(file_name)
	fig = plt.figure(figsize=(14, 8))
	name = basename(file_name)
	fig.suptitle(name, fontsize=20)
	addWaveSubplot(fig, samples, sample_rate, time_length)
	addLogSpectrogramSubplot(fig, samples, sample_rate, time_length)
	log_MSG = generateDBMelSpectrogram(samples, sample_rate)
	addDBMelSpectrogram(fig, log_MSG, sample_rate)
	mfcc, delta2_mfcc = generateMFCCs(log_MSG, 13)
	addDelta2MFCCSubplot(fig, delta2_mfcc)  
	plt.tight_layout()
	plt.subplots_adjust(top=0.90)
	if (save_dir is not None):
		plt.savefig(os.path.join(save_dir, os.path.splitext(name)[0] + ".png"))
	plt.show()


# Takes wav file path. Returns sample rate, time series as floats, and length of wav file in seconds
def readWavFloat(file_name):
	sample_rate, samples = wavfile.read(file_name)
	# convert to float to make time series samples compatible with librosa mfcc functions
	duration = len(samples) / sample_rate
	return sample_rate, samples.astype(float), duration


# Takes input time series (Samples) and sample rate. Returns frequencies, times, and log spectrogram
def log_specgram(samples, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(samples,
									fs=sample_rate,
									window='hann',
									nperseg=nperseg,
									noverlap=noverlap,
									detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# Takes spectrogram. Normalizes spectrogram, returns spectrogram, mean, and standard deviation
def normalizeSpectrogram(spectrogram):
	# normalize spectrogram
	mean = np.mean(spectrogram, axis=0)
	std = np.std(spectrogram, axis=0)
	spectrogram = (spectrogram - mean) / std
	return spectrogram, mean, std


# Takes time series and sample rate. Returns Mel Spectrogram in DB.
def generateDBMelSpectrogram(samples, sample_rate):
	S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
	# Convert to log scale (dB). We'll use the peak power (max) as reference.
	log_S = librosa.power_to_db(S, ref=np.max)
	return log_S


# Takes log mel spectrogram and the number of MFCC coefficients. Returns mfcc and second delta mfcc.
def generateMFCCs(log_MSG, num_coefficients):
	mfcc = librosa.feature.mfcc(S=log_MSG, n_mfcc=num_coefficients)
	mfcc2 = librosa.feature.delta(mfcc, order=2)
	return mfcc, mfcc2


# Takes figure, log spectrogram, and sample rate. Sets DB Mel Spectrogram subplot to figure.
def addDBMelSpectrogram(fig, log_S, sample_rate):
	ax3 = fig.add_subplot(413)
	ax3.set_title('Mel Power Spectrogram')
	ax3.set_ylabel('Hz')
	ax3.set_xlabel('Seconds')
	librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
	plt.colorbar(format='%+02.0f dB')


# Takes figure, time series, sample rate, and length of audio. Adds wave subplot to figure.
def addWaveSubplot(fig, samples, sample_rate, time_length):
	ax1 = fig.add_subplot(411)
	ax1.set_title('Raw Wave')
	ax1.set_ylabel('Amplitude')
	ax1.set_xlabel('Seconds')
	ax1.plot(np.linspace(0, time_length, len(samples)), samples)


# Takes figure, time series, sample rate, and length of audio. Adds log spectrogram subplot to figure.
def addLogSpectrogramSubplot(fig, samples, sample_rate, time_length):
	freqs, times, spectrogram = log_specgram(samples, sample_rate)
	ax2 = fig.add_subplot(412)
	ax2.imshow(spectrogram.T, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
	ax2.set_yticks(freqs[::16])
	ax2.set_xticks(np.linspace(0, time_length, 10))
	ax2.set_title('Log Spectrogram')
	ax2.set_ylabel('Hz')
	ax2.set_xlabel('Seconds')


# takes figure and delta 2 mfcc. Adds delta2 mfcc to subplot
def addDelta2MFCCSubplot(fig, delta2_mfcc):
	ax4 = fig.add_subplot(414)
	librosa.display.specshow(delta2_mfcc, x_axis='time')
	ax4.set_title(r'MFCC-$\Delta^2$')
	ax4.set_ylabel('MFCC coeffs')
	ax4.set_xlabel('Time')
	plt.colorbar()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Audio Utils')
	parser.add_argument("type", help="specify histogram or spectrogram", type=str)
	parser.add_argument("input", help="specify input path", type=str)
	parser.add_argument("output", help="specify output directory", type=str)
	args = parser.parse_args()
	if args.type == 'histogram':
		if not os.path.isdir(args.input):
			print('Usage: audio_utils.py <type> <input_path> <output_path>')
		plot_durations_histogram(args.input, args.output)
	elif args.type == 'spectrogram':
		if not os.path.isfile(args.input):
			print('Usage: audio_utils.py <type> <input_path> <output_path>')
		plot_spectrograms(args.input, args.output)
	else:
		print('Usage: audio_utils.py <type> <input_path> <output_path>')
