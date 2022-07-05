#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# TODO: Add to dockerfile
cd /workspace/espnet/tools && make moses.done
cd /workspace/espnet/egs2/st_covost2/st1

# language related
src_lang=id
tgt_lang=en

# src_nbpe=1000
# tgt_nbpe=1000
src_nbpe=300
tgt_nbpe=300
src_case=lc.rm
tgt_case=lc.rm

st_config=conf/train_st.yaml
inference_config=conf/decode_st.yaml

is_low_resource=true

if [ ${is_low_resource} = true ]; then
    speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"
else
    speed_perturb_factors="0.9 1.0 1.1"
fi

main_folder=st_covost2
# sub_folder=bpe100_no_pretrain_4gb_moses
sub_folder=bpe300_no_pretrain_2gb_moses
train_data_folder=/datasets/id_data_2gb_cleaned
test_data_folder=/datasets/test_id_data_1gb_cleaned
# data_tag=
data_tag=2gb_clean

st_exp=/mount/exp/${main_folder}/${sub_folder}/st_exp

# train_set=train${data_tag}.${src_lang}-${tgt_lang}
# train_dev=valid${data_tag}.${src_lang}-${tgt_lang}
# test_sets="test${data_tag}.${src_lang}-${tgt_lang} valid${data_tag}.${src_lang}-${tgt_lang}"

train_set=train_${data_tag}.${src_lang}-${tgt_lang}
train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
test_sets="test_${data_tag}.${src_lang}-${tgt_lang} valid_${data_tag}.${src_lang}-${tgt_lang}"

. utils/parse_options.sh || exit 1;

./st.sh \
    --stage 1 \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
    --ngpu 1 \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --use_lm false \
    --feats_type raw \
    --audio_format "wav" \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe ${src_nbpe} \
    --tgt_token_type "bpe" \
    --tgt_nbpe ${tgt_nbpe} \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@" \
    --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
    --st_exp "${st_exp}" \
    --st_stats_dir "/mount/exp/${main_folder}/${sub_folder}/st_stats"
    # --pretrained_asr "/mount/exp/java/id/pretrain_default_config_4gb/asr_exp/valid.acc.ave.pth"
    # --src_bpe_train_text "data/train.${src_lang}-${tgt_lang}/text.${src_case}.${src_lang}" \
    # --tgt_bpe_train_text "data/train.${src_lang}-${tgt_lang}/text.${tgt_case}.${tgt_lang}" \

# Show results in Markdown syntax
if [[ ${tgt_case} == "lc.rm" ]]; then
    case=lc
else
    case=${tgt_case}
fi

scripts/utils/show_translation_result.sh --case ${case} "${st_exp}" > "${st_exp}"/RESULTS.md
cat "${st_exp}"/RESULTS.md
