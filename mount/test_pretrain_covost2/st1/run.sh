#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/egs2/test_pretrain_covost2/st1

# language related
src_lang=id
tgt_lang=en

src_nbpe=1000
tgt_nbpe=1000
src_case=lc.rm
tgt_case=lc.rm

train_set=train.${src_lang}-${tgt_lang}
train_dev=valid.${src_lang}-${tgt_lang}
test_sets="test.${src_lang}-${tgt_lang} valid.${src_lang}-${tgt_lang}"

utils/spk2utt_to_utt2spk.pl data/${train_set}/spk2utt > data/${train_set}/utt2spk
utils/fix_data_dir.sh data/${train_set}

utils/spk2utt_to_utt2spk.pl data/${train_dev}/spk2utt > data/${train_dev}/utt2spk
utils/fix_data_dir.sh data/${train_dev}

for test_set in ${test_sets}; do
    utils/spk2utt_to_utt2spk.pl data/${test_set}/spk2utt > data/${test_set}/utt2spk
    utils/fix_data_dir.sh data/${test_set}
done

utils/fix_data_dir.sh dump/raw/train.id-en_sp
utils/fix_data_dir.sh dump/raw/valid.id-en
utils/fix_data_dir.sh dump/raw/test.id-en

st_config=conf/train_st.yaml
inference_config=conf/decode_st.yaml

# verify language directions
is_exist=false
is_low_resource=false
if [[ ${src_lang} == en ]]; then
    tgt_langs=de_ca_zh-CN_fa_et_mn_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
else
    lr_src_langs=it_ru_zh-CN_pt_fa_et_mn_nl_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${lr_src_langs} | tr '_' ' '); do
        if [[ ${lang} == "${src_lang}" ]]; then
            is_low_resource=true
            break
        fi
    done
    src_langs=fr_de_es_ca_it_ru_zh-CN_pt_fa_et_mn_nl_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${src_langs} | tr '_' ' '); do
        if [[ ${lang} == "${src_lang}" ]]; then
            is_exist=true
            break
        fi
    done
fi
if [[ ${is_exist} == false ]]; then
    echo "No language direction: ${src_lang} to ${tgt_lang}" && exit 1;
fi

if [ ${is_low_resource} = true ]; then
    speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"
else
    speed_perturb_factors="0.9 1.0 1.1"
fi

if [ ${src_lang} == ja ] || [ ${src_lang} == zh-CN ]; then
    src_nbpe=4000
fi

if [ ${tgt_lang} == ja ] || [ ${tgt_lang} == zh-CN ]; then
    tgt_nbpe=4000
fi

main_folder=test_pretrain_covost2
sub_folder=test_own_data

./st.sh \
    --stage 2 \
    --ngpu 1 \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --use_lm false \
    --feats_type raw \
    --audio_format "wav" \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
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
    --st_exp "/mount/exp/${main_folder}/${sub_folder}/st_exp" \
    --st_stats_dir "/mount/exp/${main_folder}/${sub_folder}/st_stats"
    # --local_data_opts "--stage 1 --src_lang ${src_lang} --tgt_lang ${tgt_lang}" \
