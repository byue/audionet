import os
from os.path import expanduser
from os.path import join
import shutil
from shutil import move
from shutil import rmtree
import subprocess
import sys
# Note: Must run infra/scripts/init-audio-dependencies
LIBROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
sys.path.insert(0, LIBROOT)
import silence_remover as sr
from pathlib import Path


ROOT = os.path.join(os.path.expanduser('~'), 'Audio_Bank')
RAW_ROOT = os.path.join(ROOT, 'Raw')
SOURCE_ROOT = os.path.join(ROOT, 'Sources')
META_ROOT = os.path.join(ROOT, 'Meta')


# returns list of names of all datasets in RAW root
def allDatasetNames():
	return [name for name in os.listdir(RAW_ROOT) \
			if os.path.isdir(os.path.join(RAW_ROOT, name))]


# for each dataset returns a list of unique source folders.
def source_to_folder_mappings(dataset_list):
	dataset_mappings = {}
	for dataset in dataset_list:
		paths = []
		dataset_path = os.path.join(RAW_ROOT, dataset)
		if dataset == 'voxforge':
			folders = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) \
				if os.path.isdir(os.path.join(dataset_path, d))]
			for folder in folders:
				sub_folders = os.listdir(folder)
				file_type = 'wav'
				if 'flac' in sub_folders:
					file_type = 'flac'
				wav_path = os.path.join(folder, file_type)
				paths.append(wav_path)
		elif dataset == 'tatoeba':
			tatoeba_audio = os.path.join(dataset_path, 'audio')
			folders = os.listdir(tatoeba_audio)
			for source in folders:
				wav_path = os.path.join(tatoeba_audio, source)
				paths.append(wav_path)
		dataset_mappings[dataset] = paths
	return dataset_mappings


# moves sources from RAW to SOURCES, renames folders based on prefixes (v for voxforge, t for tatoeba)
# and index of source, starting from 0. Deletes RAW root at the end. Input is dataset_list, a list of 
# dataset names
def move_sources(dataset_list):
	dataset_mappings = source_to_folder_mappings(dataset_list)
	source_index = 0
	for dataset in dataset_mappings.keys():
		sources = dataset_mappings[dataset]
		for old_folder in sources:
			sample_index = 0
			prefix = None
			if dataset == 'voxforge':
				prefix = 'v'
			elif dataset == 'tatoeba':
				prefix = 't'
			folder_name = prefix + str(source_index)
			source_root = os.path.join(SOURCE_ROOT, folder_name)
			if not os.path.isdir(source_root):
				os.mkdir(source_root)
			files = [os.path.join(old_folder, f) for f in os.listdir(old_folder) \
				if (f.endswith('.wav') and os.path.isfile(os.path.join(old_folder, f)))]
			for old_path in files:
				sample_name = folder_name + '-' + str(sample_index) + '.wav'
				new_path = os.path.join(source_root, sample_name)
				shutil.move(old_path, new_path)
				sample_index += 1
			source_index += 1
	shutil.rmtree(RAW_ROOT)


# Preprocesses audio in SOURCE by removing silences with VAD dependency, and chunks audio in each source
# folder by combining audio in one clip and then chopping up in 5 second chunks.
def preprocess_sources():
	source_folders = [os.path.join(SOURCE_ROOT, d) for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
	count = 1
	for source_folder in source_folders:
		print('Applying vad to ' + str(source_folder) + ", " + str(count) + '/' + str(len(source_folders)))
		remove_silences(source_folder)
		if len(os.listdir(source_folder)) == 0:
			shutil.rmtree(source_folder)
		count += 1


# remove silences from audio files in source_folder with vad. A higher aggressiveness (1 - 3) means more chopping.
def remove_silences(source_folder, VAD_aggressiveness=2):
	files = [f for f in os.listdir(source_folder) if (f.endswith('.wav') and os.path.isfile(os.path.join(source_folder, f)))]
	for f in files:
		path = os.path.join(source_folder, f)
		sr.remove_silences(VAD_aggressiveness, path)


# Chunks audio in SOURCE folder into 5 second segments
def chunk_audio():
	cmd = '~/iMic/infra/scripts/chunk-audio.sh'
	if (subprocess.call(cmd, shell=True) != 0):
		return True
	return False

# copies files from RAW to SOURCE, deletes RAW, and preprocess SOURCE files by removing silences and chunking into 5 second segments.
def main():
	move_sources(allDatasetNames())
	preprocess_sources()
	print('chunking audio')
	chunk_audio()


if __name__ == '__main__':
	main()
