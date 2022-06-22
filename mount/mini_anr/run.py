import subprocess
from clearml import Task

task = Task.init(
    project_name = 'espnet_speech_translation',
    task_name = 'test_mini',
    output_uri = 's3://experiment-logging/storage/espnet'
)

task.set_base_docker('dleongsh/espnet:202205-torch1.10-cu113-runtime')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

subprocess.run([
    './asr.sh',
    '--lang', 'en',
    '--train_set', 'train_nodev',
    '--valid_set', 'train_dev',
    '--test_sets', 'train_dev test test_seg',
    '--lm_train_text', "data/train_nodev/text" "$@",
    '--lm_exp', "/mount/exp/mini_anr/transformer/lm_exp",
    '--lm_stats_dir', "/mount/exp/mini_anr/transformer/lm_stats",
    '--asr_exp', "/mount/exp/mini_anr/transformer/asr_exp",
    '--asr_stats_dir', "/mount/exp/transformer/asr_stats",
    '--asr_config', "/mount/mini_anr/tuning/train_asr_transformer.yaml",
    '--inference_config', "/mount/mini_anr/tuning/decode_transformer.yaml"
    ])