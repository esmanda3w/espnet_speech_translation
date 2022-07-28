import os
import shutil
import subprocess
from clearml import Task, Model, Dataset
Task.add_requirements('setuptools', '59.5.0')

task_name = 'bpe800_finetune_2gb_clean'

task = Task.init(
    project_name = 'espnet_speech_translation',
    task_name = task_name,
    output_uri = 's3://experiment-logging/storage/espnet',
)

task.set_base_docker('dleongsh/espnet:202205-torch1.10-cu113-runtime')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

# download dataset
clearml_train_dataset = Dataset.get(dataset_id='e5231a33cf6349e5ab9857c904f9d891')
train_dataset_path = clearml_train_dataset.get_local_copy()

clearml_test_dataset = Dataset.get(dataset_id='8068aea0cd8c4068ad560d3946742347')
test_dataset_path = clearml_test_dataset.get_local_copy()

# # download model
# clearml_model = Model(model_id='...')
# model_path = clearml_model.get_local_copy()

# Work-around for nltk crash
shutil.copytree('../../../nltk_data', '/root/nltk_data', symlinks=True)

shutil.copytree('../../../scripts', '/scripts', symlinks=True)
shutil.copytree('../../e2e_st', '/workspace/espnet/egs2/e2e_st', symlinks=True)

subprocess.run([
    './run.sh',
    '--sub_folder', task_name,
    '--train_data_folder', train_dataset_path,
    '--test_data_folder', test_dataset_path,
    # '--pretrained_model', model_path,
    ])


output_dirs = {
    # 'lm_exp': "./exp/e2e_st/test_clearml/lm_exp",
    # 'lm_stats_dir': "./exp/e2e_st/test_clearml/lm_stats",
    'st_exp': f"./exp/e2e_st/{task_name}/st_exp",
    'st_stats_dir': f"./exp/e2e_st/{task_name}/st_stats",
}

for key in output_dirs.keys():
    dirx = output_dirs[key]
    for root, _, filenames in os.walk(dirx):
        for filename in filenames:

            # to avoid uploading all the epoch weights
            if filename.endswith('epoch.pth'):
                continue

            filepath = os.path.join(root, filename)
            task.upload_artifact(f'{key}_{filename}', artifact_object=filepath)
