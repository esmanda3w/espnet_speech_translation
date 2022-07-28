#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

##### GENERAL CONFIG #####

# Language related
src_lang=id
tgt_lang=en

# Experiment folder related
cascade_main_folder=cascade
cascade_sub_folder=lm_finetune_6gb_clean_asr_bpe1000_batchbin500000_6gb_moses_mt

# ASR experiment folder
asr_main_folder="java"
asr_package_model_name=lm_finetune_6gb_clean_asr

# MT experiment folder
mt_main_folder="mt_test"
mt_package_model_name=bpe1000_batchbin500000_6gb_moses_mt

# Data related
train_data_folder=/datasets/id_data_2gb_cleaned       # not actually used but required to run
test_data_folder=/datasets/test_id_data_1gb_cleaned
data_tag=infer

skip_asr=false
skip_mt=false

##########################

. utils/parse_options.sh || exit 1;

#########################
##### ASR INFERENCE #####
#########################

if ! "${skip_asr}"; then

    cd /workspace/espnet/egs2/${asr_main_folder}/asr1

    ##### SETTING UP ASR #####

    bpe_model=bpe_unigram1000
    inference_asr_model=valid.acc.best.pth
    use_lm=true
    lm_model=valid.loss.ave.pth

    # Copying ASR config file and model
    mkdir -p /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/
    cp /models/${asr_package_model_name}/config.yaml /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/
    cp /models/${asr_package_model_name}/${inference_asr_model} /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/

    # Reconfigure feats_stats path in config.py
    python3 /scripts/modify_asr_config.py \
        --config_file_path  /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/config.yaml \
        --feats_stats_path /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_stats/train/feats_stats.npz

    # Creating model's token list
    mkdir -p data/token_list/${bpe_model}/
    cp /models/${asr_package_model_name}/tokens.txt data/token_list/${bpe_model}/

    # Copying the bpe model
    cp /models/${asr_package_model_name}/${bpe_model}/bpe.model data/token_list/${bpe_model}/

    if "${use_lm}"; then
        mkdir -p /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/lm_exp/
        cp /models/${asr_package_model_name}/lm/config.yaml /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/lm_exp/
        cp /models/${asr_package_model_name}/lm/${lm_model} /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/lm_exp/
    fi

    ##########################

    # Decode config related
    inference_config=conf/decode_asr.yaml

    # Data related 
    asr_train_set=train_${data_tag}
    asr_train_dev=valid_${data_tag}
    asr_test_sets="test_${data_tag}"

    # Run stage 1-4, 9-13
    for stage in 1 2 3 4 9 10 12 13; do
        ./asr.sh \
            --stage ${stage} \
            --stop_stage ${stage} \
            --local_data_opts "--src_lang ${src_lang} --tgt_lang ${tgt_lang} --train_data_folder ${train_data_folder} --test_data_folder ${test_data_folder} --data_tag ${data_tag}" \
            --ngpu 1 \
            --use_lm ${use_lm} \
            --token_type bpe \
            --nbpe 1000 \
            --feats_type raw \
            --inference_config "${inference_config}" \
            --train_set "${asr_train_set}" \
            --valid_set "${asr_train_dev}" \
            --test_sets "${asr_test_sets}" \
            --lm_exp "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/lm_exp" \
            --lm_stats_dir "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/lm_stats" \
            --asr_exp "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp" \
            --asr_stats_dir "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_stats" \
            --inference_asr_model ${inference_asr_model}

    done
fi

########################
##### MT INFERENCE #####
########################
if ! "${skip_mt}"; then

    cd /workspace/espnet/tools && make moses.done
    cd /workspace/espnet/egs2/${mt_main_folder}/mt1

    ##### SETTING UP MT #####

    inference_mt_model=valid.acc.ave.pth

    mkdir -p /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/
    cp /models/${mt_package_model_name}/config.yaml /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/
    cp /models/${mt_package_model_name}/${inference_mt_model} /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/

    #########################

    # Tokenization related
    src_nbpe=1000
    tgt_nbpe=1000
    src_case=lc.rm
    tgt_case=lc.rm

    # Decode config related
    inference_config=conf/decode_mt.yaml

    # Cascade ASR output into MT input
    asr_inference_text=/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/*/*/text
    local_data_opts="--stage 0 \
                    --src_lang ${src_lang} \
                    --tgt_lang ${tgt_lang} \
                    --train_data_folder ${train_data_folder} \
                    --test_data_folder ${test_data_folder} \
                    --data_tag ${data_tag}
                    --asr_inference_text ${asr_inference_text}"
    
    # Path of experiment folder
    cascade_mt_exp=/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp

    # Data related
    mt_train_set=train_${data_tag}.${src_lang}-${tgt_lang}
    mt_train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
    mt_test_sets="test_${data_tag}.${src_lang}-${tgt_lang}"

    # Run stages 1-2, 11-12
    for stage in 1 2 11 12; do
        ./mt.sh \
            --stage ${stage} \
            --stop_stage ${stage} \
            --local_data_opts "${local_data_opts}" \
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
            --inference_config ${inference_config} \
            --train_set ${mt_train_set} \
            --valid_set ${mt_train_dev} \
            --test_sets ${mt_test_sets} \
            --src_bpe_train_text "data/${mt_train_set}/text.${src_case}.${src_lang}" \
            --tgt_bpe_train_text "data/${mt_train_set}/text.${tgt_case}.${tgt_lang}" \
            --mt_exp "${cascade_mt_exp}" \
            --mt_stats_dir "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_stats" \
            --inference_mt_model ${inference_mt_model}
    done

    # Show results in Markdown syntax
    if [[ ${tgt_case} == "lc.rm" ]]; then
        case=lc
    else
        case=${tgt_case}
    fi

    scripts/utils/show_translation_result.sh --case ${case} "${cascade_mt_exp}" > "${cascade_mt_exp}"/RESULTS.md
    cat "${cascade_mt_exp}"/RESULTS.md

fi
