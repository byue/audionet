import os
from os.path import join
import random
from random import shuffle
from random import sample


ROOT = os.path.join(os.path.expanduser('~'), 'Audio_Bank')
META_ROOT = os.path.join(ROOT, 'Meta')
SOURCE_ROOT = os.path.join(ROOT, 'Sources')
OUTPUT_ROOT = os.path.join(ROOT, 'Output')


# Input is experiment_name, string of the experiment to be created.
# Creates directory structure of Meta and Output folders
def create_experiment_directory(experiment_name):
	experiment_root = os.path.join(META_ROOT, experiment_name)
	train_folder = os.path.join(experiment_root, 'Train')
	validation_folder = os.path.join(experiment_root, 'Validation')
	test_folder = os.path.join(experiment_root, 'Test')
	output_folder = os.path.join(OUTPUT_ROOT, experiment_name)
	if os.path.isdir(experiment_root):
		print('error, experiment dir exists already')
		return
	os.mkdir(experiment_root)
	os.mkdir(train_folder)
	os.mkdir(validation_folder)
	os.mkdir(test_folder)
	os.mkdir(output_folder)


# Splits up unique sources randomly into train, test, and validation sources with specified percentages. 
# Input is string of experiment name, and percentages of train, validation, and test sets which must add up to 100. 
# Creates Sources csv files for each set (see directory_structure.txt). Returns the number of sources in each set.
def setup_train_validation_test(experiment_name, train_percentage=80, validation_percentage=10, test_percentage=10):
	assert train_percentage + validation_percentage + test_percentage == 100
	source_path = os.path.join(META_ROOT, experiment_name)
	assert os.path.isdir(source_path)
	train_path = os.path.join(source_path, 'Train')
	assert os.path.isdir(train_path)
	val_path = os.path.join(source_path, 'Validation')
	assert os.path.isdir(val_path)
	test_path = os.path.join(source_path, 'Test')
	assert os.path.isdir(test_path)
	source_names = [d for d in os.listdir(SOURCE_ROOT)
		if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
	shuffle(source_names)
	num_train = int(len(source_names) * (float(train_percentage) / 100))
	num_validation = int(len(source_names) * (float(validation_percentage) / 100))
	num_test = len(source_names) - num_train - num_validation
	train_sources = source_names[0:num_train]
	validation_sources = source_names[num_train:num_train + num_validation]
	test_sources = source_names[num_train + num_validation:]
	with open(os.path.join(train_path, 'Train_Sources.csv'),'w') as f:
		for source in train_sources:
			f.write(source + '\n')
	with open(os.path.join(val_path, 'Validation_Sources.csv'),'w') as f:
		for source in validation_sources:
			f.write(source + '\n')
	with open(os.path.join(test_path, 'Test_Sources.csv'),'w') as f:
		for source in test_sources:
			f.write(source + '\n')
	return num_train, num_validation, num_validation


# Generates mix csv files for each set for an experiment. Default is 2-source mixes.
# Num_combinations specifies how many combinations of mixes are in each set, created from the set sources.
def generate_mixes(experiment_name, num_sources=2, num_combinations={'Train': 12800, 'Validation': 1600, 'Test': 1600}):
	assert set(num_combinations.keys()) == set(['Train', 'Validation', 'Test'])
	for set_type in num_combinations.keys():
		generate_set_mix(experiment_name, set_type, num_sources, num_combinations[set_type])


# Generates pairings for each dataset_type. Each source is weighted by a randomly generated SNR between min_weight
# and max_weight. Writes pairings and weights to CSV (see directory_structure.txt).
# Params:
# experiment_name (string): name of the experiment.
# dataset_type (string): Train, Validation, or Test.
# num_sources (int): Number of sources in each mix
# num_combinations (int): Number of combinations for each set.
# min_weight (int): The mininum SNR weight in the randomly generated SNR weight per source in the mix.
# max_weight (int): The maximum SNR weight in the randomly generated SNR weight per source in the mix.
def generate_set_mix(experiment_name, dataset_type, num_sources, num_combinations, min_weight=0, max_weight=5):
	source_path = os.path.join(META_ROOT, experiment_name)
	assert os.path.isdir(source_path)
	assert (dataset_type == 'Train' or dataset_type == 'Validation' or dataset_type == 'Test')
	subset_path = os.path.join(source_path, dataset_type)
	assert os.path.isdir(subset_path)
	source_file_path = os.path.join(subset_path, dataset_type + '_Sources.csv')
	sources = None
	with open(source_file_path, 'r') as f:
		sources = f.read().splitlines()
	mix_file_path = os.path.join(subset_path, dataset_type + '_Mix.csv')
	with open(mix_file_path, 'w') as f:
		header = ''
		for i in range(0, num_sources):
			if i != 0:
				header += ','
			header += ('Source-' + str(i) + ',Weight-' + str(i))
		f.write(header + '\n')
		for i in range(0, num_combinations):
			combo = random.sample(sources, num_sources)
			line = ''
			first = True
			for source in combo:
				root = os.path.join(SOURCE_ROOT, source)
				samples = [f for f in os.listdir(root) if (f.endswith('.wav') and os.path.isfile(os.path.join(root, f)))]
				sample = random.choice(samples)
				name = os.path.splitext(sample)[0]
				weight = random.randint(min_weight, max_weight)
				if not first:
					line += ','
				line += (name + ',' + str(weight))
				if first:
					first = False
			f.write(line + '\n')


# Creates experiment directories and source/mix CSV files
def main():
	experiment_name = 'test1'
	create_experiment_directory(experiment_name)
	setup_train_validation_test(experiment_name)
	generate_mixes(experiment_name)


if __name__ == '__main__':
	main()
