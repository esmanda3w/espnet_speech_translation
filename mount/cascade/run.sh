#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

##### GENERAL CONFIG #####

src_lang=id
tgt_lang=en

cascade_main_folder=cascade
cascade_sub_folder=test_inference

asr_main_folder="java"
asr_exp_folder="java/id/pretrain_default_config_4gb"

mt_main_folder="mt_test"
mt_exp_folder="mt_test/test_own_data"

train_data_folder=/datasets/id_data_2gb_cleaned
test_data_folder=/datasets/test_id_data_1gb_cleaned
data_tag=inference

##########################

#########################
##### ASR INFERENCE #####
#########################

cd /workspace/espnet/egs2/${asr_main_folder}/asr1

bpe_model=bpe_unigram1000

##### SETTING UP ASR #####

# Copying ASR config file and model
mkdir -p /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/
cp /mount/exp/${asr_exp_folder}/asr_exp/config.yaml /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/
cp /mount/exp/${asr_exp_folder}/asr_exp/valid.acc.best.pth /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/

# Creating model's token list
pyscripts/utils/make_token_list_from_config.py /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/config.yaml
mkdir -p data/token_list/${bpe_model}/
cp /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/tokens.txt data/token_list/${bpe_model}/

# TODO: change this to the packaged asr model dir
cp /models/jv_openslr35/bpe.model data/token_list/${bpe_model}/
##########################

inference_config=conf/decode_asr.yaml

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
        --inference_args "--batch_size 1" \
        --use_lm false \
        --token_type bpe \
        --nbpe 1000 \
        --feats_type raw \
        --inference_config "${inference_config}" \
        --train_set "${asr_train_set}" \
        --valid_set "${asr_train_dev}" \
        --test_sets "${asr_test_sets}" \
        --asr_exp "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp" \
        --asr_stats_dir "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_stats" \
        --inference_asr_model "valid.acc.best.pth" \

    if [[ ${stage} -eq 10 ]]; then
        python3 /scripts/modify_asr_config.py \
            --config_file_path  /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/config.yaml \
            --feats_stats_path /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_stats/train/feats_stats.npz
    fi

done

########################
##### MT INFERENCE #####
########################

cd /workspace/espnet/tools && make moses.done
cd /workspace/espnet/egs2/${mt_main_folder}/mt1

##### SETTING UP MT #####

mkdir -p /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/
cp /mount/exp/${mt_exp_folder}/mt_exp/config.yaml /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/
cp /mount/exp/${mt_exp_folder}/mt_exp/valid.acc.best.pth /mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp/

#########################

src_nbpe=1000
tgt_nbpe=1000
src_case=lc.rm
tgt_case=lc.rm

inference_config=conf/decode_mt.yaml

asr_inference_text=/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/asr_exp/*/*/text
local_data_opts="--stage 0 \
                 --src_lang ${src_lang} \
                 --tgt_lang ${tgt_lang} \
                 --train_data_folder ${train_data_folder} \
                 --test_data_folder ${test_data_folder} \
                 --data_tag ${data_tag} \
                 --asr_inference_text ${asr_inference_text}"
cascade_mt_exp=/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_exp

mt_train_set=train_${data_tag}.${src_lang}-${tgt_lang}
mt_train_dev=valid_${data_tag}.${src_lang}-${tgt_lang}
mt_test_sets="test_${data_tag}.${src_lang}-${tgt_lang}"

# Run stages 1-2, 11-12 (need use scoring fix)
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
        --inference_config "${inference_config}" \
        --train_set "${mt_train_set}" \
        --valid_set "${mt_train_dev}" \
        --test_sets "${mt_test_sets}" \
        --src_bpe_train_text "data/${mt_train_set}/text.${src_case}.${src_lang}" \
        --tgt_bpe_train_text "data/${mt_train_set}/text.${tgt_case}.${tgt_lang}" \
        --mt_exp "${cascade_mt_exp}" \
        --mt_stats_dir "/mount/exp/${cascade_main_folder}/${cascade_sub_folder}/mt_stats" \
        --inference_mt_model "valid.acc.best.pth"
done

# Show results in Markdown syntax
if [[ ${tgt_case} == "lc.rm" ]]; then
    case=lc
else
    case=${tgt_case}
fi

scripts/utils/show_translation_result.sh --case ${case} "${cascade_mt_exp}" > "${cascade_mt_exp}"/RESULTS.md
cat "${cascade_mt_exp}"/RESULTS.md
