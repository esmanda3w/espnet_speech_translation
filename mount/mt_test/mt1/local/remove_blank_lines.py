import os
import sys
import argparse

from numpy import split

def parse_args():
    parser = argparse.ArgumentParser(
        description="Removing blank lines predicted by the ASR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # og path
    parser.add_argument("--src_language",    type=str, help="Source language of the machine translation task")
    parser.add_argument("--tgt_language",    type=str, help="Target language of the machine translation task")
    parser.add_argument("--data_folder",     type=str, help="Path to folder containing data files")
    
    return parser.parse_args(sys.argv[1:])

class BlankLinesRemover:
    def __init__(self, src_lang, tgt_lang, data_folder):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_folder = data_folder

    def remove(self):
        blank_lines = self._find_blank_lines()

        files = ["text", f"text.{self.src_lang}", f"text.{self.tgt_lang}", "wav.scp"]
        for file in files:
            original_file = open(os.path.join(self.data_folder, file), "r")
            new_file = open("temp.txt", "w")

            for line in original_file:
                utt_id = line.strip().split(' ')[0]
                if utt_id not in blank_lines:
                    new_file.write(line)

            original_file.close()
            new_file.close()

            # replace file with original name
            os.replace("temp.txt", os.path.join(self.data_folder, file))

        original_file = open(os.path.join(self.data_folder, "spk2utt"), "r")
        line = original_file.readline()

        for blank_line in blank_lines:
            line.replace(f" {blank_line}", "")

        new_file = open("temp.txt", "w")
        new_file.write(line)

        original_file.close()
        new_file.close()

        # replace file with original name
        os.replace("temp.txt", os.path.join(self.data_folder, "spk2utt"))

    def _find_blank_lines(self):
        blank_lines = []

        text_file = open(os.path.join(self.data_folder, "text"), "r")

        for line in text_file:
            split_line = line.strip().split(' ')
            # Remove blank lines
            if len(split_line) == 1:
                blank_lines.append(split_line[0])
        
        return blank_lines

if __name__ == "__main__":
    args = parse_args()
    rem = BlankLinesRemover(
        args.src_language,
        args.tgt_language,
        args.data_folder,
        )

    rem.remove()
