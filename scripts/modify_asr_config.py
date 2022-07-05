import sys
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Converting the predictions of an ASR model to the input format of an MT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # og path
    parser.add_argument("--config_file_path",   type=str, help="Path to the experiment directory where the predictions are stored (eg. xxx/yyy/asr_exp)")
    parser.add_argument("--feats_stats_path",   type=str, help="Path to the experiment directory where the predictions are stored (eg. xxx/yyy/asr_exp)")
    
    return parser.parse_args(sys.argv[1:])

class ModifyASRConfig:
    def __init__(self, config_file_path, feats_stats_path):
        self.config_file_path = config_file_path
        self.feats_stats_path = feats_stats_path

    def modify(self):
        original_config_file = open(self.config_file_path, "r")
        new_config_file = open("./config.yaml", "w")
        
        for line in original_config_file:
            print(line)
            if "stats_file:" in line:
                new_line = line.split("stats_file:", 1)[0] + f"stats_file: {self.feats_stats_path}\n"
                print(line)
                print(new_line)
                new_config_file.write(new_line)
            else:
                new_config_file.write(line)
        
        original_config_file.close()
        new_config_file.close()

        shutil.move("./config.yaml", self.config_file_path)

if __name__ == "__main__":
    args = parse_args()
    
    mod = ModifyASRConfig(
        args.config_file_path,
        args.feats_stats_path,
        )

    mod.modify()
