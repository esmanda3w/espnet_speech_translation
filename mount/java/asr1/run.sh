#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/egs2/java/asr1

lid=false # whether to use language id as additional label

main_folder="java/cascade"
sub_folder="inference"
data_folder=/datasets/id_en
data_tag=inference

train_set=train_${data_tag}
train_dev=valid_${data_tag}
test_sets="test_${data_tag}"

train_model=jv_openslr35
bpe_model=bpe_unigram1000

# Setting up required files
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
pyscripts/utils/make_token_list_from_config.py /models/${train_model}/config.yaml
mkdir -p data/token_list/${bpe_model}/
cp /models/${train_model}/tokens.txt data/token_list/${bpe_model}/
cp /models/${train_model}/bpe.model data/token_list/${bpe_model}/bpe.model
mkdir -p /mount/exp/${main_folder}/${sub_folder}/asr_stats/train
cp /models/${train_model}/feats_stats.npz /mount/exp/${main_folder}/${sub_folder}/asr_stats/train

# Run stage 1-4, 9-13
./asr.sh \
    --stage 1 \
    --stop_stage 1 \
    --ngpu 1 \
    --nj 80 \
    --inference_nj 256 \
    --gpu_inference true \
    --inference_args "--batch_size 1" \
    --use_lm false \
    --token_type bpe \
    --nbpe 1000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@" \
    --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
    --asr_exp "/mount/exp/${main_folder}/${sub_folder}/asr_exp" \
    --asr_stats_dir "/mount/exp/${main_folder}/${sub_folder}/asr_stats" \
    --pretrained_model "/models/${train_model}/valid.acc.best.pth" \
    --inference_asr_model "valid.acc.best.pth" \
    --ignore_init_mismatch "true"
