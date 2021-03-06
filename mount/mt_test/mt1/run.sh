#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/tools && make moses.done
cd /workspace/espnet/egs2/mt_test/mt1

##### GENERAL CONFIG #####

# Language related
src_lang=id
tgt_lang=en

# Experiment folder related
main_folder=mt_test
sub_folder=bpe1000_batchbin500000_6gb_moses

# Data related
train_data_folder=/datasets/id_data_6gb_cleaned
test_data_folder=/datasets/test_id_data_1gb_cleaned
data_tag=6gb_clean

train_set=train_${data_tag}.${src_lang}-${tgt_lang}
train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
test_sets="test_${data_tag}.${src_lang}-${tgt_lang}"

# Tokenization related
src_nbpe=1000
tgt_nbpe=1000
src_case=lc.rm
tgt_case=lc.rm

# Train and decode config related
mt_config=conf/train_mt.yaml
inference_config=conf/decode_mt.yaml

# Path of experiment folder
mt_exp=/mount/exp/${main_folder}/${sub_folder}/mt_exp

# MT model for decoding
inference_mt_model=valid.acc.ave.pth

# Package model related
package_model=true
package_model_name=bpe1000_batchbin500000_6gb_moses_mt

. utils/parse_options.sh || exit 1;

##########################

./mt.sh \
    --stage 1 \
    --stop_stage 12 \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
    --ngpu 1 \
    --use_lm false \
    --feats_type raw \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_nbpe ${src_nbpe} \
    --tgt_nbpe ${tgt_nbpe} \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --mt_config "${mt_config}" \
    --inference_config "${inference_config}" \
    --train_set ${train_set} \
    --valid_set ${train_dev} \
    --test_sets ${test_sets} \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
    --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
    --mt_exp "${mt_exp}" \
    --mt_stats_dir "/mount/exp/${main_folder}/${sub_folder}/mt_stats" \
    --inference_mt_model "{inference_mt_model}"

##### SHOW RESULTS #####

if [[ ${tgt_case} == "lc.rm" ]]; then
    case=lc
else
    case=${tgt_case}
fi

scripts/utils/show_translation_result.sh --case ${case} "${mt_exp}" > "${mt_exp}"/RESULTS.md
cat "${mt_exp}"/RESULTS.md

########################

##### PACKAGE MODEL #####

if "${package_model}"; then

    if [ -z "${package_model_name}" ]; then
        log "Error: Package model name is required."
        exit 2
    fi

    model_dir=/models/${package_model_name}
    mkdir -p ${model_dir}

    # Copy model's config and pretrained model (.pth)
    cp /mount/exp/${main_folder}/${sub_folder}/mt_exp/config.yaml ${model_dir}
    cp /mount/exp/${main_folder}/${sub_folder}/mt_exp/${inference_mt_model} ${model_dir}

fi

#########################
