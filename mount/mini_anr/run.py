import subprocess
from clearml import Task
Task.add_requirements('setuptools', '59.5.0')

task = Task.init(
    project_name = 'espnet_speech_translation',
    task_name = 'test_mini',
    output_uri = 's3://experiment-logging/storage/espnet'
)

task.set_base_docker('dleongsh/espnet:202205-torch1.10-cu113-runtime')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

subprocess.run(['./run.sh'])