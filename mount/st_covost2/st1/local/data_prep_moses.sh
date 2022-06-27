#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src-lang> <tgt-lang>"
    echo "e.g.: $0 source_lang target_lang"
    exit 1;
fi

src_lang=$1
tgt_lang=$2

for set in train valid test; do
    dst=data/dump/${set}.${src_lang}-${tgt_lang}
    mkdir -p ${dst} || exit 1;

    src=data/${set}.${src_lang}-${tgt_lang}/text.${src_lang}
    tgt=data/${set}.${src_lang}-${tgt_lang}/text.${tgt_lang}

    cut -f 2- -d " " ${src} > ${dst}/${src_lang}.org
    cut -f 2- -d " " ${tgt} > ${dst}/${tgt_lang}.org
    cut -f 1 -d " " ${src} > ${dst}/reclist.${src_lang}
    cut -f 1 -d " " ${tgt} > ${dst}/reclist.${tgt_lang}

    for lang in ${src_lang} ${tgt_lang}; do
        lang_trim="$(echo "${lang}" | cut -f 1 -d '-')"

        # normalize punctuation
        if [ ${lang} = ${src_lang} ]; then
            lowercase.perl < ${dst}/${lang}.org > ${dst}/${lang}.org.lc
            # NOTE: almost all characters in transcription on CommonVoice is truecased
            normalize-punctuation.perl -l ${lang_trim} < ${dst}/${lang}.org.lc > ${dst}/${lang}.norm
        else
            normalize-punctuation.perl -l ${lang_trim} < ${dst}/${lang}.org > ${dst}/${lang}.norm
        fi

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        for case in lc.rm lc tc; do
            # tokenization
            tokenizer.perl -l ${lang_trim} -q < ${dst}/${lang}.norm.${case} > ${dst}/${lang}.norm.${case}.tok

            paste -d " " ${dst}/reclist.${lang} ${dst}/${lang}.norm.${case}.tok | sort > ${dst}/text.${case}.${lang}
        done

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done

    # extract common lines
    comm -12 <(sort ${dst}/reclist.${src_lang}) <(sort ${dst}/reclist.${tgt_lang}) > ${dst}/reclist

    # Copy stuff into its final locations [this has been moved from the format_data script]
    reduce_data_dir.sh ${dst} ${dst}/reclist data/${set}.${src_lang}-${tgt_lang}
    for case in lc.rm lc tc; do
        cp ${dst}/text.${case}.${src_lang} data/${set}.${src_lang}-${tgt_lang}
        cp ${dst}/text.${case}.${tgt_lang} data/${set}.${src_lang}-${tgt_lang}
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} \
         text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" data/${set}.${src_lang}-${tgt_lang}

    echo "$0: successfully prepared data in ${dst}"
done
