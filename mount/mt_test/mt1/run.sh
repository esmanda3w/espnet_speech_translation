#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/tools && make moses.done
cd /workspace/espnet/egs2/mt_test/mt1

# language related
src_lang=id
tgt_lang=en

src_nbpe=1000
tgt_nbpe=1000
src_case=lc.rm
tgt_case=lc.rm

mt_config=conf/train_mt.yaml
inference_config=conf/decode_mt.yaml

main_folder=mt_test
sub_folder=test_own_data
data_folder=/datasets/id_en
data_tag=""

train_set=train_${data_tag}.${src_lang}-${tgt_lang}
train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
test_sets="test_${data_tag}.${src_lang}-${tgt_lang} valid_${data_tag}.${src_lang}-${tgt_lang}"

./mt.sh \
    --stage 10 \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --tgt_lang ${tgt_lang} --data_folder ${data_folder}" \
    --ngpu 1 \
    --use_lm false \
    --feats_type raw \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe ${src_nbpe} \
    --tgt_token_type "bpe" \
    --tgt_nbpe ${tgt_nbpe} \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --mt_config "${mt_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
    --mt_exp "/mount/exp/${main_folder}/${sub_folder}/mt_exp" \
    --mt_stats_dir "/mount/exp/${main_folder}/${sub_folder}/mt_stats"