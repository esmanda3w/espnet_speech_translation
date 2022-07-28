#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cwd=$PWD

cd /workspace/espnet/tools && make moses.done
cd /workspace/espnet/egs2/e2e_st/st1

##### GENERAL CONFIG #####

local_run=false   # run script locally or on ClearML

# Language related
src_lang=id
tgt_lang=en

# Experiment folder related
main_folder=e2e_st
sub_folder=mtloss_bpe1000_finetune_pretrain_6gb_clean

# Data related
train_data_folder=/datasets/id_data_6gb_cleaned
test_data_folder=/datasets/test_id_data_1gb_cleaned
data_tag=6gb_clean

train_set=train_${data_tag}.${src_lang}-${tgt_lang}
train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
test_sets="test_${data_tag}.${src_lang}-${tgt_lang} valid_${data_tag}.${src_lang}-${tgt_lang}"

# Tokenization related
src_nbpe=1000
tgt_nbpe=1000
src_case=lc.rm
tgt_case=lc.rm

# Speed perturbation related
speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"

# Train and decode config related
st_config=conf/train_st.yaml
inference_config=conf/decode_st.yaml

# Initialise encoder and decoder from ASR and MT modules
st_args="--init_param \
        /mount/exp/java/id/lm_finetune_6gb_clean/asr_exp/valid.acc.best.pth:encoder:encoder \
        /mount/exp/mt_test/encoutput256_bpe1000_batchbin500000_6gb_moses/mt_exp/valid.acc.best.pth:decoder:decoder"

. utils/parse_options.sh || exit 1;

##########################

# Path of experiment folder
[ "${local_run}" ] && cwd=/mount
st_exp=${cwd}/exp/${main_folder}/${sub_folder}/st_exp

./st.sh \
    --stage 1 \
    --stop_stage 13 \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
    --ngpu 1 \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --use_lm false \
    --feats_type raw \
    --audio_format "wav" \
    --token_joint false \
    --src_lang "${src_lang}" \
    --tgt_lang "${tgt_lang}" \
    --src_nbpe "${src_nbpe}" \
    --tgt_nbpe "${tgt_nbpe}" \
    --src_case "${src_case}" \
    --tgt_case "${tgt_case}" \
    --st_config "${st_config}" \
    --st_args "${st_args}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --lm_exp "${cwd}/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "${cwd}/exp/${main_folder}/${sub_folder}/lm_stats" \
    --st_exp ${st_exp} \
    --st_stats_dir "${cwd}/exp/${main_folder}/${sub_folder}/st_stats"

##### SHOW RESULTS #####

if [ ${tgt_case} == "lc.rm" ]; then
    case=lc
else
    case=${tgt_case}
fi

scripts/utils/show_translation_result.sh --case ${case} "${st_exp}" > "${st_exp}"/RESULTS.md
cat "${st_exp}"/RESULTS.md

########################

