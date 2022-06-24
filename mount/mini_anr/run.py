import os
import subprocess
from clearml import Task, Model, Dataset
Task.add_requirements('setuptools', '59.5.0')

task = Task.init(
    project_name = 'espnet_speech_translation',
    task_name = 'test_mini',
    output_uri = 's3://experiment-logging/storage/espnet'
)

task.set_base_docker('dleongsh/espnet:202205-torch1.10-cu113-runtime')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

# download dataset
clearml_dataset = Dataset.get(dataset_id='...')
dataset_path = clearml_dataset.get_local_copy()

# download model
clearml_model = Model(model_id='...')
model_path = clearml_model.get_local_copy()

subprocess.run([
    './run.sh',
    '--main_folder', dataset_path,
    '--pretrained_model', model_path,
    ])

output_dirs = {
    'lm_exp': "./exp/mini_anr/transformer/lm_exp",
    'lm_stats_dir': "./exp/mini_anr/transformer/lm_stats",
    'asr_exp': "./exp/mini_anr/transformer/asr_exp",
    'asr_stats_dir': "./exp/transformer/asr_stats",
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


