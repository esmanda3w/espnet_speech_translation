import os
import shutil
import subprocess
from clearml import Task, Model, Dataset
Task.add_requirements('setuptools', '59.5.0')

task = Task.init(
    project_name = 'espnet_speech_translation',
    task_name = 'test_st_covost2',
    output_uri = 's3://experiment-logging/storage/espnet',
)

task.set_base_docker('dleongsh/espnet:202205-torch1.10-cu113-runtime')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

# download dataset
clearml_dataset = Dataset.get(dataset_id='d1ef7778c97f404dadf1eb77eb8cd069')
dataset_path = clearml_dataset.get_local_copy()
print(dataset_path)
print(os.listdir(dataset_path))

# # download model
# clearml_model = Model(model_id='...')
# model_path = clearml_model.get_local_copy()

data_tag = "test_clearml"

shutil.copytree('../../st_covost2', '/workspace/espnet/egs2/st_covost2', symlinks=True)

subprocess.run([
    './run.sh',
    '--data_folder', dataset_path,
    '--data_tag', data_tag,
    # '--pretrained_model', model_path,
    ])

output_dirs = {
    'lm_exp': "./exp/st_covost2/test_clearml/lm_exp",
    'lm_stats_dir': "./exp/st_covost2/test_clearml/lm_stats",
    'st_exp': "./exp/st_covost2/test_clearml/st_exp",
    'st_stats_dir': "./exp/st_covost2/test_clearml/st_stats",
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
