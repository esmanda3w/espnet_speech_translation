#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

train_set=
valid_set=
test_sets=

set -e
set -u
set -o pipefail

. utils/parse_options.sh

utils/spk2utt_to_utt2spk.pl data/${train_set}/spk2utt > data/${train_set}/utt2spk
utils/fix_data_dir.sh data/${train_set}

utils/spk2utt_to_utt2spk.pl data/${valid_set}/spk2utt > data/${valid_set}/utt2spk
utils/fix_data_dir.sh data/${valid_set}

for test_set in ${test_sets}; do
    utils/spk2utt_to_utt2spk.pl data/${test_set}/spk2utt > data/${test_set}/utt2spk
    utils/fix_data_dir.sh data/${test_set}
done
