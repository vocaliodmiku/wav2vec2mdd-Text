# env
export KALDI_ROOT=/disk2/plk/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

path=/disk2/plk/wav2vec/experiments/text_condition
compute-wer --text --mode=present ark:$path/annotation.txt ark:$path/hypo.txt > per || exit 1;

# ark:text1.txt ark:text2.txt ark,t:alignment.txt
align-text ark:$path/ref.txt  ark:$path/annotation.txt ark,t:- | wer_per_utt_details.pl > ref_human_detail
align-text ark:$path/annotation.txt  ark:$path/hypo.txt ark,t:- | wer_per_utt_details.pl > human_our_detail
align-text ark:$path/ref.txt  ark:$path/hypo.txt ark,t:- | wer_per_utt_details.pl > ref_our_detail
python3 ins_del_sub_cor_analysis.py