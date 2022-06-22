#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cd /workspace/espnet/egs2/commonvoice/asr1

lang=ru # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk

train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set="${train_dev} test_$(echo ${lang} | tr - _)"

# asr_config=conf/tuning/train_asr_conformer5.yaml
asr_config=/mount/commonvoice/tuning/train_asr_conformer5.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

if [[ "zh" == *"${lang}"* ]]; then
  nbpe=2500
elif [[ "fr" == *"${lang}"* ]]; then
  nbpe=350
elif [[ "es" == *"${lang}"* ]]; then
  nbpe=235
else
  nbpe=150
fi

./asr.sh \
    --stage 1 \
    --ngpu 1 \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@" \
    --lm_exp "/mount/exp/commonvoice/half_batchbin_double_accumgrad_ru/lm_exp" \
    --lm_stats_dir "/mount/exp/commonvoice/half_batchbin_double_accumgrad_ru/lm_stats" \
    --asr_exp "/mount/exp/commonvoice/half_batchbin_double_accumgrad_ru/asr_exp" \
    --asr_stats_dir "/mount/exp/commonvoice/half_batchbin_double_accumgrad_ru/asr_stats"
