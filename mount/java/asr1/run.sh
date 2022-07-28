#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd /workspace/espnet/egs2/java/asr1

##### PRETRAINED MODEL SETUP #####

pretrain_model=jv_openslr35
bpe_model=bpe_unigram1000
inference_asr_model=valid.acc.best.pth

# Creating model's token list
pyscripts/utils/make_token_list_from_config.py /models/${pretrain_model}/config.yaml
mkdir -p data/token_list/${bpe_model}/
cp /models/${pretrain_model}/tokens.txt data/token_list/${bpe_model}/

# Copying the bpe model
cp /models/${pretrain_model}/${bpe_model}/bpe.model data/token_list/${bpe_model}/

##################################

##### GENERAL CONFIG #####

# Language related
src_lang=id
tgt_lang=en

# Experiment folder related
main_folder="java/id"
sub_folder="lm_finetune_6gb_clean"

# Data related
train_data_folder=/datasets/id_data_6gb_cleaned
test_data_folder="/datasets/test_id_data_1gb_cleaned"
data_tag=6gb_cleaned

train_set=train_${data_tag}
train_dev=valid_${data_tag}
test_sets="test_${data_tag}"

# Train and decode config related
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

# LM related
use_lm=true

# Package model related
package_model=true
package_model_name=lm_finetune_6gb_clean_asr

. utils/parse_options.sh || exit 1;

##########################

# Run stage 1-4, 6-8 (LM), 9-13
for stage in 1 2 3 4 6 7 8 9 10 11 12 13; do
    ./asr.sh \
        --stage ${stage} \
        --stop_stage ${stage} \
        --local_data_opts "--src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
        --ngpu 1 \
        --use_lm ${use_lm} \
        --token_type bpe \
        --nbpe 1000 \
        --feats_type raw \
        --asr_config ${asr_config} \
        --inference_config ${inference_config} \
        --train_set ${train_set} \
        --valid_set ${train_dev} \
        --test_sets ${test_sets} \
        --lm_train_text "data/${train_set}/text" \
        --lm_exp "/mount/exp/${main_folder}/${sub_folder}/lm_exp" \
        --lm_stats_dir "/mount/exp/${main_folder}/${sub_folder}/lm_stats" \
        --asr_exp "/mount/exp/${main_folder}/${sub_folder}/asr_exp" \
        --asr_stats_dir "/mount/exp/${main_folder}/${sub_folder}/asr_stats" \
        --inference_asr_model "${inference_asr_model}" \
        --pretrained_model "/models/${pretrain_model}/${inference_asr_model}" \
        --ignore_init_mismatch "true"
done

##### PACKAGE MODEL #####

if "${package_model}"; then

    if [ -z "${package_model_name}" ]; then
        log "Error: Package model name is required."
        exit 2
    fi

    model_dir=/models/${package_model_name}
    mkdir -p ${model_dir}/${bpe_model}

    # Copy model's config and pretrained model (.pth)
    cp /mount/exp/${main_folder}/${sub_folder}/asr_exp/config.yaml ${model_dir}
    cp /mount/exp/${main_folder}/${sub_folder}/asr_exp/${inference_asr_model} ${model_dir}

    # Creating model's token list
    pyscripts/utils/make_token_list_from_config.py ${model_dir}/config.yaml

    # Copy model's bpe.model
    cp data/token_list/${bpe_model}/bpe.model ${model_dir}/${bpe_model}

    if [ "${use_lm}" ]; then
        lm_model=valid.loss.ave.pth
        mkdir -p ${model_dir}/lm
        cp /mount/exp/${main_folder}/${sub_folder}/lm_exp/config.yaml ${model_dir}/lm
        cp /mount/exp/${main_folder}/${sub_folder}/lm_exp/${lm_model} ${model_dir}/lm
    fi

fi

#########################
