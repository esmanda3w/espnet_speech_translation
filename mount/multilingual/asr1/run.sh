#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/egs2/multilingual/asr1

lid=true # whether to use language id as additional label

main_folder=multilingual
sub_folder=pretrain_default_2gb

train_set=train_2gb
train_dev=valid_2gb
test_set=test_2gb

train_model=open_li52
bpe_model=bpe_unigram7000

# Setting up required files
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
pyscripts/utils/make_token_list_from_config.py /models/${train_model}/config.yaml
mkdir -p data/token_list/${bpe_model}/
cp /models/${train_model}/tokens.txt data/token_list/${bpe_model}/
cp /models/${train_model}/bpe.model data/token_list/${bpe_model}/bpe.model
mkdir -p /mount/exp/${main_folder}/${sub_folder}/asr_stats/train
cp /models/${train_model}/feats_stats.npz /mount/exp/${main_folder}/${sub_folder}/asr_stats/train
# nlsyms_txt=data/local/nlsyms.txt

# Run stage 1, 9-11

./asr.sh \
    --stage 11 \
    --stop_stage 40 \
    --ngpu 1 \
    --nj 80 \
    --inference_nj 256 \
    --use_lm false \
    --token_type bpe \
    --nbpe 7000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --local_data_opts "--train_set ${train_set} --valid_set ${train_dev} --test_sets ${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@" \
    --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
    --asr_exp "/mount/exp/${main_folder}/${sub_folder}/asr_exp" \
    --asr_stats_dir "/mount/exp/${main_folder}/${sub_folder}/asr_stats" \
    --pretrained_model "/models/open_li52/valid.acc.ave_10best.pth"
    # --download_model "https://zenodo.org/record/4509663/files/asr_train_asr_transformer_e40_raw_bpe7000_valid.acc.ave.zip?download=1"
    # --local_data_opts "--langs ${langs} --stage 0 --lid ${lid} --nlsyms_txt ${nlsyms_txt}" \
    # --nlsyms_txt "${nlsyms_txt}" \
