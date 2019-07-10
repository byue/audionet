import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import audio_dataset as ad
import time
import itertools
import matplotlib.pyplot as plt
import math
import librosa


# data_dict tags
SOURCES_TAGS = ['sources_stft_mag', 'sources_stft_theta']
MIX_TAGS = ['mix_stft_mag', 'mix_stft_theta']


ROOT = os.path.join(os.path.expanduser('~'), 'Audio_Bank')
OUTPUT_ROOT = os.path.join(ROOT, 'Output')


# each batch is several utterances, each with time vs frequency stft
EPOCHS = 200
BATCH_SIZE = 256
NUM_SOURCES = 2


# Stopping conditions
MIN_EPOCH_STOP = 150


# LSTM layer
NUM_FEATURES = 129
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5


# Linear layer
INPUT_DIMENSION_LINEAR = 2 * HIDDEN_SIZE
OUTPUT_DIMENSIONS = NUM_FEATURES * NUM_SOURCES


# Step size
INITIAL_LEARNING_RATE = 2e-5


# STFT indices for readability
MAGNITUDE_INDEX = 0
THETA_INDEX = 1


# for isft
HOP_LENGTH = 128


# Note: magnitude normalization does not help
# See https://arxiv.org/pdf/1708.09588.pdf and https://arxiv.org/pdf/1703.06284.pdf
# for PIT model outline/params.
class PitModel(torch.nn.Module):
	def __init__(self):
		super(PitModel, self).__init__()
		self.lstm = nn.LSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT, bidirectional=True)
		self.linear = nn.Linear(in_features=INPUT_DIMENSION_LINEAR, out_features=OUTPUT_DIMENSIONS)

	def forward(self, input):
		predicted, _ = self.lstm(input) 
		predicted = predicted.contiguous().view(-1, predicted.size(-1)) # flatten timesteps with batchsize, (batchsize*timesteps x features)
		predicted = self.linear(predicted)
		predicted = F.relu(predicted)
		predicted = predicted.view(-1, input.size(1), predicted.size(-1)) # unflatten, output is (batchsize, timesteps, output_dimension) : (8, variable, 258); output_dim is size of concatenated masks
		return predicted


# Trains the experiment (CSV files must be created from audio_preprocessing). Saves and returns the trained model.
# Plots losses of train/validation set every epoch. Prints train/validation loss per epoch. Learning rate is multiplied
# by 0.7 when training epoch loss increases. Input to model is STFT magnitude frames of mix.
# Output is masks. We do element-wise multiplication of masks and mix to get each predicted source. Loss is calculated between predicted source
# and actual source. We calculate losses for each combination of target-source/mask pairing and use the combination with the mininum loss.
# Early terminationL: Training stops after MIN_EPOCH_STOP when learning rate decreases below 1e-10 or if validation loss per epoch increases.
#
# Params:
# experiment_name (string): name of the experiment
# epochs (int): number of epochs
# batch_size (int): size of one batch
# tensorboard (boolean): whether tensorboard is enabled or not
def train(experiment_name, epochs=EPOCHS, batch_size=BATCH_SIZE, tensorboard=False):
	with open('log.txt', 'w') as log:
		model = prepare_model(PitModel())
		optimizer = opt.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
		train_dataloader, _ = get_data_loader('test1', 'Train', batch_size)
		validation_dataloader, _ = get_data_loader('test1', 'Validation', batch_size)
		prevTrainLoss = sys.float_info.max
		currTrainLoss = None
		prevValidationLoss = sys.float_info.max
		currValidationLoss = None
		train_losses = []
		validation_losses = []
		for e in range(0, epochs):
			terminate_cond1 = currTrainLoss is not None and currTrainLoss > prevTrainLoss and adjust_lr(optimizer, prevTrainLoss, currTrainLoss, 1e-10)
			terminate_cond2 = currValidationLoss is not None and currValidationLoss > prevValidationLoss and (e + 1) > MIN_EPOCH_STOP
			if terminate_cond1 or terminate_cond2:
				log.write('----------------------------------------------\n')
				log.write('Early Termination\n')
				log.flush()
				torch.save(model, 'pit_model-' + str(experiment_name) + '.pt')
				return model
			if currTrainLoss is not None:
				prevTrainLoss = currTrainLoss
			if currValidationLoss is not None:
				prevValidationLoss = currValidationLoss
			validation_start = time.time()
			currValidationLoss = validate_epoch(model, validation_dataloader)
			validation_losses.append(currValidationLoss)
			plot_losses(validation_losses, 'Validation Curve')
			currTrainLoss = train_epoch(model, train_dataloader, optimizer)
			train_end = time.time()
			train_losses.append(currTrainLoss)
			plot_losses(train_losses, 'Training_Curve')
			log.write('Epoch ' + str(e + 1) + '/' + str(EPOCHS) + ', Validation Loss: ' + str(currValidationLoss) + ', Train Loss: ' + str(currTrainLoss) + ', Epoch Time(s): ' + str(train_end - validation_start) + '\n')
			log.flush()
		torch.save(model, 'pit_model-' + experiment_name + '.pt')
		return model


# Runs model on test set dataloader. Saves audio files of mix, actual sources, 
# and predicted sources to OUTPUT folder. Returns average loss for the test dataset.
def test(model, data_loader):
	return eval_epoch(model, data_loader, True)


# Validates model on validation dataloader. Returns average loss in the epoch.
def validate_epoch(model, data_loader):
	return eval_epoch(model, data_loader, False)


# Returns average loss in the epoch for the data_loader and model.
# save_files specifies whether we save audio files of mix and sources to OUTPUT (predicted/actual).
def eval_epoch(model, data_loader, save_files):
	sum_loss = 0.0
	for b, batch in enumerate(data_loader):
		batch_start = time.time()
		data_dict = prepare_tensors(batch)
		output = model(data_dict['mix_stft_mag'])
		pit_loss = get_pit_loss(output, data_dict, b, save_files)
		sum_loss += pit_loss.item()
	return sum_loss / len(data_loader)


# Given a model and data_loader trains model on one epoch. If log is True we print the batch number
# and loss in that batch.
def train_epoch(model, data_loader, optimizer, log=False):
	sum_loss = 0.0
	with open("log.txt", "a") as log:
		for b, batch in enumerate(data_loader):
			batch_start = time.time()
			data_dict = prepare_tensors(batch)
			output = model(data_dict['mix_stft_mag'])
			pit_loss = get_pit_loss(output, data_dict, b, save_files=False)
			if log:
				log.write(str(b + 1) + '/' + str(len(data_loader)) + ' loss: ' + str(pit_loss.item()) + '\n')
				log.flush()
			model.zero_grad()
			pit_loss.backward()
			optimizer.step()
			sum_loss += pit_loss.item()
		return sum_loss / len(data_loader)


# Takes STFT frames of magnitude, and STFT frames of theta. Combines STFT frames into 
# complex numbers.
def get_complex_stft(stft_mag, stft_theta):
	real = torch.mul(stft_mag, torch.cos(stft_theta)).data.numpy()
	imaginary_tensor = 1j * torch.mul(stft_mag, torch.sin(stft_theta))
	if torch.cuda.is_available():
		imaginary_tensor = imaginary_tensor.cpu()
	imaginary = imaginary_tensor.data.numpy()
	return np.add(real, imaginary)


# writes wav files for batch given stft_batch, experiment name, batch number, sample type (mix, predicted, actual), 
# and source number (speaker number)
def write_stft_batch(stft_batch, experiment_name, batch_num, sample_type, source_num):
	experiment_path = os.path.join(OUTPUT_ROOT, experiment_name)
	assert os.path.isdir(experiment_path)
	sample_index = batch_num * BATCH_SIZE
	for i in range(0, BATCH_SIZE):
		sample_folder = os.path.join(experiment_path, 'sample-' + str(sample_index))
		if not os.path.isdir(sample_folder):
			os.mkdir(sample_folder)
		ts = librosa.istft(stft_batch[i].T, hop_length=HOP_LENGTH, center=True)
		path = os.path.join(sample_folder, 'sample-' + str(sample_index) + '-' + sample_type + '-' + str(source_num) +'.wav')
		librosa.output.write_wav(path, ts, 16000)
		sample_index += 1


# Returns PIT L2 Loss given output tensor, data_dict of mag/theta stft's for source and mixes, and batch number.
# save_files specifies whether we will convert to wav and save the mix, target sources, and predicted sources.
def get_pit_loss(output, data_dict, batch_num, save_files=False):
	criterion = nn.MSELoss()
	masks = []
	true_stfts = []
	for i in range(0, NUM_SOURCES):
		start = i * NUM_FEATURES
		mask = output[:, :, start:(start + NUM_FEATURES)]
		masks.append(mask)
		true_stfts.append((data_dict['sources_stft_mag'][i], data_dict['sources_stft_theta'][i]))
	mix_stft_mag = data_dict['mix_stft_mag']
	mix_stft_theta = data_dict['mix_stft_theta']		
	# get all combinations of masks and sources
	minComboLoss = None
	min_pairings = None
	combinations = [zip(x, masks) for x in itertools.permutations(true_stfts, len(masks))]
	for pairings in combinations:
		# sum up losses from each pair in combo
		pairings = list(pairings)
		sigmaSourceLoss = None
		for pair in pairings:
			# Loss with PSA regularization
			true_stft_mag = pair[0][MAGNITUDE_INDEX]
			true_stft_theta = pair[0][THETA_INDEX]
			mask = pair[1]
			phase_delta = torch.sub(mix_stft_theta, true_stft_theta)
			psa_regularization = torch.cos(phase_delta)
			regularized_truth = torch.mul(true_stft_mag, psa_regularization)
			predicted_stft_mag = torch.mul(mix_stft_mag, mask)
			loss = criterion(predicted_stft_mag, regularized_truth)
			if sigmaSourceLoss is None:
				sigmaSourceLoss = loss
			else:
				sigmaSourceLoss += loss
		# only retain min loss from combinations
		if minComboLoss is None or minComboLoss.item() > sigmaSourceLoss.item():
			minComboLoss = sigmaSourceLoss
			min_pairings = pairings
	if save_files:
		mix_stft = get_complex_stft(mix_stft_mag, mix_stft_theta)
		write_stft_batch(mix_stft, 'test1', batch_num, 'mix', 0)
		source_num  = 0
		for pair in min_pairings:
			mask = pair[1]
			predicted_stft_mag = torch.mul(mix_stft_mag, mask)
			true_stft_mag = pair[0][MAGNITUDE_INDEX]
			true_stft_theta = pair[0][THETA_INDEX]
			predicted_stft = get_complex_stft(predicted_stft_mag, mix_stft_theta)
			write_stft_batch(predicted_stft, 'test1', batch_num, 'predicted', source_num)
			true_stft = get_complex_stft(true_stft_mag, true_stft_theta)
			write_stft_batch(true_stft, 'test1', batch_num, 'source', source_num)
			source_num += 1
	return minComboLoss


# Given an experiment name, the set_Type (Train, Validation, Test), and the size of the batch,
# returns a dataloader for that dataset.
def get_data_loader(experiment_name, set_type, batchsize=BATCH_SIZE):
	audio_dataset = ad.AudioDataset(experiment_name, set_type, transform=transforms.Compose([ad.ToTimeSeries(), ad.ToSTFT(), ad.ToTensor()]))
	return DataLoader(audio_dataset, batch_size=batchsize, shuffle=True, num_workers=4, collate_fn=ad.Collate()), len(audio_dataset)


# Use cuda with model if possible
def prepare_model(model):
	with open("log.txt", "a") as log:
		if torch.cuda.is_available():
			log.write('Using GPU...\n')
			log.flush()
			model = nn.DataParallel(model).cuda()
		else:
			log.write('Using CPU...\n')
			log.flush()
		return model


# Converts tensors to variables. Uses variables with cuda if possible.
def prepare_tensors(data_dict):
	using_gpu = torch.cuda.is_available()
	for tag in MIX_TAGS:
		data_dict[tag] = Variable(data_dict[tag])
		if using_gpu:
			data_dict[tag] = data_dict[tag].cuda()
	for tag in SOURCES_TAGS:
		for source in range(0, len(data_dict[tag])):
			data_dict[tag][source] = Variable(data_dict[tag][source])
			if using_gpu:
				data_dict[tag][source] = data_dict[tag][source].cuda()
	return data_dict


# Given a list of tensors, calculates the mean and standard deviation
def getMeanSTDV(inputs):
	composite = inputs[0]
	for i in range(1, len(inputs)):
		composite = torch.cat([composite, inputs[i]], dim=0)
	composite = composite.contiguous().view(-1, composite.size(-1))
	return torch.mean(composite, 0), torch.std(composite, 0)


# normalize the input tensor given mean and stddv tensors.
def normalize_frequencies(input, mean, stddv):
	timesteps = input.size(1)
	norm_input = input.contiguous().view(-1, input.size(-1))
	norm_input = torch.div(torch.sub(norm_input, mean), stddv)
	norm_input = norm_input.view(-1, timesteps, norm_input.size(-1))
	return norm_input


# denormalize input tensor given mean and stddv
def denormalize_frequencies(input, mean, stddv):
	timesteps = input.size(1)
	norm_input = input.contiguous().view(-1, input.size(-1))
	norm_input = torch.add(torch.mul(input, stddv), mean)
	norm_input = norm_input.view(-1, timesteps, norm_input.size(-1))
	return norm_input


# If current loss (float) is greater than previous loss (float), multiply learning
# rate (float) by 0.7. Returns true if learning rate is below floor (float).
def adjust_lr(optimizer, prevLoss, currLoss, floor):
	if (currLoss > prevLoss):
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.7
			if (param_group['lr'] < floor):
				return True
	return False


# Returns current learning rate (float).
def get_lr(optimizer):
	return optimizer.param_groups[0]['lr']		


# Given a list of losses per epoch, saves plot of epoch vs. loss. 
def plot_losses(losses, file_name):
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.title(file_name)
	x = list(range(0, len(losses)))
	y = losses
	plt.plot(x, y)
	plt.savefig('training_curves/' + file_name + '.png')


def main():
	experiment_name = 'test1'
	trained_model = train(experiment_name)
	test_dataloader, _ = get_data_loader(experiment_name, 'Test')
	test_set_loss = test(trained_model, test_dataloader)
	with open("log.txt", "a") as log:
		log.write('Average test set loss: ' + str(test_set_loss) + '\n')
		log.flush()


if __name__ == '__main__':
	main()
