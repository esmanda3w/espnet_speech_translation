#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General configuration
stage=0
stop_stage=100
src_lang=id
tgt_lang=en
train_data_folder=
test_data_folder=
data_tag=
train_data_split=80
valid_data_split=20

. utils/parse_options.sh || exit 1;

if [ -z ${data_tag} ]; then 
    log "Error: data_tag required"
    exit 2
fi

for set in train valid test; do
    if [ -d "data/${set}_${data_tag}.${src_lang}-${tgt_lang}" ]; then
        log "Error: Directory data/${set}_${data_tag}.${src_lang}-${tgt_lang} exists. Start from stage 2 to skip data preparation stage."
        exit 1
    fi
done

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Generating Kaldi style data directories from jtubespeech data"

    python3 local/data_prep.py \
        --src_language ${src_lang} \
        --tgt_language ${tgt_lang} \
        --annotated_data_filepath ${train_data_folder} \
        --data_tag ${data_tag} \
        --train_data_split ${train_data_split} \
        --validation_data_split ${valid_data_split} \
        --test_data_split 0

    python3 local/data_prep.py \
        --src_language ${src_lang} \
        --tgt_language ${tgt_lang} \
        --annotated_data_filepath ${test_data_folder} \
        --data_tag ${data_tag} \
        --train_data_split 0 \
        --validation_data_split 0 \
        --test_data_split 100 \
        --only_test_prep True

    for set in train valid test; do
        utils/spk2utt_to_utt2spk.pl data/${set}_${data_tag}/spk2utt > data/${set}_${data_tag}/utt2spk
        utils/fix_data_dir.sh data/${set}_${data_tag}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Renaming directories"
    for set in train valid test; do
        mv data/${set}_${data_tag} data/${set}_${data_tag}.${src_lang}-${tgt_lang}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Text Preprocessing with Moses (tokenization, case, punctuation marks etc.)"
    local/data_prep_moses.sh --src_lang ${src_lang} --tgt_lang ${tgt_lang} --data_tag ${data_tag}
fi
