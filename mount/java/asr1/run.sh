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

cwd=$PWD

cd /workspace/espnet/egs2/java/asr1

pretrain_model=jv_openslr35
bpe_model=bpe_unigram1000
inference_asr_model=valid.acc.best.pth

##### SETTING UP ASR #####

# Creating model's token list
pyscripts/utils/make_token_list_from_config.py /models/${pretrain_model}/config.yaml
mkdir -p data/token_list/${bpe_model}/
cp /models/${pretrain_model}/tokens.txt data/token_list/${bpe_model}/

# Copying the bpe model
cp /models/${pretrain_model}/${bpe_model}/bpe.model data/token_list/${bpe_model}/

##########################

src_lang=id
tgt_lang=en

is_low_resource=true

if [ ${is_low_resource} = true ]; then
    speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"
else
    speed_perturb_factors="0.9 1.0 1.1"
fi

main_folder="java/id"
sub_folder="lowresspeedperturb_finetune_2gb_clean"
train_data_folder=/datasets/id_data_2gb_cleaned
test_data_folder=/datasets/test_id_data_1gb_cleaned
data_tag=2gb_clean

train_set=train_${data_tag}
train_dev=valid_${data_tag}
test_sets="test_${data_tag}"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

package_model=false
package_model_name=

local_run=false

. utils/parse_options.sh || exit 1;

if "${local_run}"; then
    cwd=/mount
fi

# Run stage 1-4, 9-13
for stage in 1 2 3 4 9 10 11 12 13; do
# for stage in 2 3 4 9 10 11 12 13; do
    ./asr.sh \
        --stage ${stage} \
        --stop_stage ${stage} \
        --local_data_opts "--src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
        --ngpu 1 \
        --speed_perturb_factors "${speed_perturb_factors}" \
        --use_lm false \
        --token_type bpe \
        --nbpe 1000 \
        --feats_type raw \
        --asr_config ${asr_config} \
        --inference_config ${inference_config} \
        --train_set ${train_set} \
        --valid_set ${train_dev} \
        --test_sets ${test_sets} \
        --lm_train_text "data/${train_set}/text" \
        --lm_exp "${cwd}/exp/${main_folder}/${sub_folder}/lm_exp" \
        --lm_stats_dir "${cwd}/exp/${main_folder}/${sub_folder}/lm_stats" \
        --asr_exp "${cwd}/exp/${main_folder}/${sub_folder}/asr_exp" \
        --asr_stats_dir "${cwd}/exp/${main_folder}/${sub_folder}/asr_stats" \
        --pretrained_model "/models/${pretrain_model}/${inference_asr_model}" \
        --inference_asr_model "${inference_asr_model}" \
        --ignore_init_mismatch "true"
done

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

fi
