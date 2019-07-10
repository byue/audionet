import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import os
from os.path import isfile
import librosa
import sys
from scipy.io.wavfile import write
import acoustics


BANK_ROOT = os.path.join(os.path.expanduser('~'), 'Audio_Bank')
META_ROOT = os.path.join(BANK_ROOT, 'Meta')
SOURCE_ROOT = os.path.join(BANK_ROOT, 'Sources')


SAMPLING_RATE = 16000
# STFT constants
FRAMESHIFT = 128
WINDOW_SIZE = 256

# See tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# AudioDataset represents a class for transforming wav files -> time series -> stft frames -> tensors 
# experiment_name (string): name of experiment
# set_type (string): Train, Test, or Validation
class AudioDataset(Dataset):
	def __init__(self, experiment_name, set_type, transform=None):
		assert set_type in ('Train', 'Test', 'Validation')
		self.set_type = set_type
		self.experiment_name = experiment_name
		set_root = os.path.join(META_ROOT, experiment_name)
		set_root = os.path.join(set_root, set_type)
		mix_path = os.path.join(set_root, set_type + '_Mix.csv')
		self.meta_frame = pd.read_csv(mix_path)
		self.transform = transform

	def __len__(self):
		return len(self.meta_frame)

	def __getitem__(self, idx):
		sample = self.meta_frame.iloc[idx].as_matrix()
		if self.transform:
			sample = self.transform(sample)
		return sample


# Returns a dictionary containing sources, a list of source time series, and
# mix, a time series of sources summed with respective SNR weights.
class ToTimeSeries(object):
	def __call__(self, sample):
		i = 0
		amps = []
		weights = []
		shape = None
		while i < len(sample) - 1:
			name = str(sample[i])
			weight = int(sample[i + 1])
			sample_path = os.path.join(os.path.join(SOURCE_ROOT, name.split('-')[0]), name + '.wav')
			amp = self.__get_audio_time_series(sample_path, SAMPLING_RATE)
			amps.append(amp)
			weights.append(weight)
			if shape is None or amp.shape[0] < shape[0]:
				shape = amp.shape
			i += 2
		min_len = shape[0]
		for i in range(0, len(amps)):
			amps[i] = amps[i][:min_len]
		mix = acoustics.Signal(np.zeros(shape), 16000)
		for i in range(0, len(amps)):
			mix += ((np.power(10, weights[i] / 20)) * amps[i])
		return {'sources_ts': amps, 'mix_ts': mix}

	def __get_audio_time_series(self, path, sampling_rate):
		if not os.path.isfile(path) or not path.endswith('.wav'):
			return None
		return librosa.load(path, sr=sampling_rate)[0]


# sources_stft_mag and theta are a list of numpy arrays of stft frames for each source
# mix_stft_mag is stft frames for the mix
class ToSTFT(object):
	def __call__(self, sample):
		sources = sample['sources_ts']
		mix = sample['mix_ts']
		sources_stft_mag = []
		sources_stft_theta = []
		for source in sources:
			ts = np.transpose(self.__get_stft(source))
			sources_stft_mag.append(np.abs(ts))
			sources_stft_theta.append(np.angle(ts))
		mix_stft = np.transpose(self.__get_stft(mix))
		mix_stft_mag = np.abs(mix_stft)
		mix_stft_theta = np.angle(mix_stft)
		return {'sources_stft_mag': sources_stft_mag, 'sources_stft_theta': sources_stft_theta, \
			'mix_stft_mag': mix_stft_mag, 'mix_stft_theta': mix_stft_theta}

	def __get_stft(self, time_series):
		return librosa.core.stft(time_series, n_fft=WINDOW_SIZE, hop_length=FRAMESHIFT, center=True)


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, sample):
		sources_stft_mag = sample['sources_stft_mag']
		sources_stft_theta = sample['sources_stft_theta']
		mix_stft_mag = sample['mix_stft_mag']
		mix_stft_theta = sample['mix_stft_theta']
		for i in range(0, len(sources_stft_mag)):
			sources_stft_mag[i] = torch.from_numpy(sources_stft_mag[i])
			sources_stft_theta[i] = torch.from_numpy(sources_stft_theta[i])
		mix_stft_mag = torch.from_numpy(mix_stft_mag)
		mix_stft_theta = torch.from_numpy(mix_stft_theta)
		return {'sources_stft_mag': sources_stft_mag, 'sources_stft_theta': sources_stft_theta, \
			'mix_stft_mag': mix_stft_mag, 'mix_stft_theta': mix_stft_theta}

# collate function is for reformatting batch.
class Collate(object):
	"""
	Takes fixed length stfts from 5 second time series. Reorganizes output as a dictionary of tags.
	"""
	# input: batch is a list of dictionaries for each batch
	# output: dictionary of tags: mix_stft_mag, mix_stft_theta, sources_stft_mag, sources_stft_theta
	# mixes are dimension (batch_size, timesteps, num_frequencies). Sources are a list of arrays with 
	# dimension: (batch_size, timesteps, num_frequencies). (batch_size, 626, 129) for 5 seconds and given window sizes.
	def collate(self, batch):
		dict_output = {}
		batch_size = len(batch)
		timesteps = batch[0]['mix_stft_mag'].shape[0]
		num_frequencies = batch[0]['mix_stft_mag'].shape[1]
		for tag in ['mix_stft_mag', 'mix_stft_theta']:
			dict_output[tag] = torch.zeros((batch_size, timesteps, num_frequencies))
			for i in range(0, len(batch)):
				dict_output[tag][i] = batch[i][tag]
		for tag in ['sources_stft_mag', 'sources_stft_theta']:
			batched_sources = []
			num_sources = len(batch[0][tag])
			for source_index in range(0, num_sources):
				batched_source = torch.zeros((batch_size, timesteps, num_frequencies))
				for i in range(0, batch_size):
					batched_source[i] = batch[i][tag][source_index]
				batched_sources.append(batched_source)
			dict_output[tag] = batched_sources
		return dict_output

	def __call__(self, batch):
		return self.collate(batch)
