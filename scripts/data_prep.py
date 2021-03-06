import os
import sys
import argparse
from pathlib import Path
import random
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing the scraped jtubespeech data to follow Kaldi style data directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # og path
    parser.add_argument("--src_language",               type=str, help="Language code of the source language for the transcript")
    parser.add_argument("--tgt_language",               type=str, help="Language code of the target language for the transcript")
    parser.add_argument("--annotated_data_filepath",    type=str, help="Path of the annotated_data folder to be written in wav.scp")
    parser.add_argument("--data_tag",                   type=str, default="", help="Optional tag to differentiate data folder from data folders of other experiments")
    parser.add_argument("--train_data_split",           type=str, help="Percentage of dataset to be used for training")
    parser.add_argument("--validation_data_split",      type=str, help="Percentage of dataset to be used for validation")
    parser.add_argument("--test_data_split",            type=str, help="Percentage of dataset to be used for testing")
    parser.add_argument("--only_test_prep",             type=bool, default=False, help="Only prepare test dataset")
    
    return parser.parse_args(sys.argv[1:])


'''
KALDI DATA GENERATOR
This class will help to process the data files generated in annotated_data to follow the Kaldi style data directory.
Note that the train-validation-test split is based on a random split and the resulting split will not match the
specified percentages exactly.
'''
class KaldiDataGenerator:
    def __init__(self, src_language, tgt_language, annotated_data_filepath, data_tag,
                 train_data_split, validation_data_split, test_data_split, only_test_prep):
        assert train_data_split + validation_data_split + test_data_split == 100, 'The three percentages should sum to 100'

        self.src_language = src_language
        self.tgt_language = tgt_language
        self.annotated_data_filepath = annotated_data_filepath
        self.train_data_split = train_data_split
        self.validation_data_split = validation_data_split
        self.test_data_split = test_data_split
        self.only_test_prep = only_test_prep
        self.train_folder = f'train_{data_tag}'
        self.valid_folder = f'valid_{data_tag}'
        self.test_folder = f'test_{data_tag}'

        random.seed(42)
        

    def generate_files(self):
        # if os.path.isdir('./data'):
        #     raise Exception("Data directory already exists! Skipping data prep")

        Path(f'./data/{self.train_folder}').mkdir(parents=True, exist_ok=True)
        Path(f'./data/{self.valid_folder}').mkdir(parents=True, exist_ok=True)
        Path(f'./data/{self.test_folder}').mkdir(parents=True, exist_ok=True)
        # Path('./data/videos').mkdir(parents=True, exist_ok=True)
        
        self._create_text_file()
        self._create_spk2utt_file()
        self._create_wav_file()


    def _create_text_file(self):
        train_text_file = open(f'./data/{self.train_folder}/text','a')
        valid_text_file = open(f'./data/{self.valid_folder}/text','a')
        test_text_file = open(f'./data/{self.test_folder}/text','a')

        train_src_text_file = open(f'./data/{self.train_folder}/text.{self.src_language}','a')
        valid_src_text_file = open(f'./data/{self.valid_folder}/text.{self.src_language}','a')
        test_src_text_file = open(f'./data/{self.test_folder}/text.{self.src_language}','a')

        train_tgt_text_file = open(f'./data/{self.train_folder}/text.{self.tgt_language}','a')
        valid_tgt_text_file = open(f'./data/{self.valid_folder}/text.{self.tgt_language}','a')
        test_tgt_text_file = open(f'./data/{self.test_folder}/text.{self.tgt_language}','a')

        for root, dirs, files in os.walk(self.annotated_data_filepath):
            for file in files:
                if file.endswith(f'{self.src_language}.trans.txt'):
                    src_transcript = open(os.path.join(root, file),'r')
                    
                    # Construct target filename from source filename
                    tgt_filename = file.replace(f'_{self.src_language}', f'_{self.tgt_language}')
                    tgt_transcript = open(os.path.join(root, tgt_filename),'r')

                    # Skip videos where the number of lines in the 2 transcripts does not match
                    if sum(1 for line in src_transcript) != sum(1 for line in tgt_transcript):
                        continue

                    src_transcript = open(os.path.join(root, file),'r')
                    tgt_transcript = open(os.path.join(root, tgt_filename),'r')

                    for src_line, tgt_line in zip(src_transcript, tgt_transcript):
                        # Remove blank lines
                        if len(src_line.strip().split(' ')) == 1 or len(tgt_line.strip().split(' ')) == 1:
                            continue

                        # Remove lines with numbers
                        if self._has_numbers(' '.join(src_line.split(' ')[1:])) \
                            or self._has_numbers(' '.join(tgt_line.split(' ')[1:])):
                            continue

                        # # Because not all wav files are in the subset uploaded to clearml
                        # wav_filename = src_line.replace(f'_{self.src_language}', '').split(' ')[0] + '.wav'
                        # if not os.path.exists(os.path.join(root, wav_filename)):
                        #     continue

                        # # Subset dataset
                        # half = random.randint(0, 99)
                        # if half < 33:
                        #     continue
                        
                        # Remove language tag from the utterance id
                        src_line = src_line.replace(f'_{self.src_language}', '')
                        tgt_line = tgt_line.replace(f'_{self.tgt_language}', '')

                        split_chance = random.randint(0,99)
                        if split_chance < self.train_data_split:
                            train_text_file.write(src_line)
                            train_src_text_file.write(src_line)
                            train_tgt_text_file.write(tgt_line)
                        elif split_chance < self.train_data_split + self.validation_data_split:
                            valid_text_file.write(src_line)
                            valid_src_text_file.write(src_line)
                            valid_tgt_text_file.write(tgt_line)
                        else:
                            test_text_file.write(src_line)
                            test_src_text_file.write(src_line)
                            test_tgt_text_file.write(tgt_line)
                        
                        # self._copy_wav_file(src_line)
        
        print('Done: Creating text files')

    def _has_numbers(self, inputString):
        return any(char.isdigit() for char in inputString)
    
    def _copy_wav_file(self, line):
        utt_id = line[:16]
        # Extract enclosing folder name (first 8 chars of utterance id)
        folder_containing_wav_file = utt_id[:8]
        Path(f'data/videos/{folder_containing_wav_file}').mkdir(parents=True, exist_ok=True)
        shutil.copy(f'{self.annotated_data_filepath}/{folder_containing_wav_file}/{utt_id}.wav', 
                    f'data/videos/{folder_containing_wav_file}/')

    def _create_spk2utt_file(self):
        # List of every utt_id in the data
        utt_id_list = ''
        data_types = [self.train_folder, self.valid_folder, self.test_folder]
        for data_type in data_types:
            text_file = open(f'./data/{data_type}/text','r')

            for line in text_file:
                # Extract utt_id (first 16 chars) from text file
                utt_id_list += f' {line[:16]}'

            spk2utt_file = open(f'./data/{data_type}/spk2utt','w')
            spk2utt_file.write(f'dummy_spk_id{utt_id_list}\n')
            utt_id_list = ''

        print('Done: Creating spk2utt file')


    def _create_wav_file(self):
        if self.only_test_prep:
            data_types = [self.test_folder]
        else:
            data_types = [self.train_folder, self.valid_folder, self.test_folder]

        for data_type in data_types:
            text_file = open(f'./data/{data_type}/text','r')
            wav_file = open(f'./data/{data_type}/wav.scp','w')

            for line in text_file:
                # Extract utt_id (first 16 chars) from text file
                utt_id = line[:16]
                # Extract enclosing folder name (first 8 chars of utterance id)
                folder_containing_wav_file = utt_id[:8]

                wav_filepath = f'{self.annotated_data_filepath}/{folder_containing_wav_file}/{utt_id}.wav'
                wav_file.write(f'{utt_id} {wav_filepath}\n')
        
        print('Done: Creating wav file')

if __name__ == "__main__":
    args = parse_args()
    gen = KaldiDataGenerator(args.src_language,
                             args.tgt_language,
                             args.annotated_data_filepath, 
                             args.data_tag, 
                             int(args.train_data_split),
                             int(args.validation_data_split),
                             int(args.test_data_split),
                             args.only_test_prep,
                            )

    # try:
    gen.generate_files()
    # except Exception as e:
    #     print("Warning: ", e)
