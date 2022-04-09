#!/disk2/plk/python/bin/
# Using wav2vec2.0 to test libirpseech TEST-CLEAN data
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export KALDI_ROOT=/path/to/kaldi

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

MODEL_PATH=$(pwd)
checkpoint=$MODEL_PATH/checkpoint_best.pt
python3 result.py wav2vec2mdd_text/data test $checkpoint || exit 1;

align-text ark:$MODEL_PATH/result/ref.txt  ark:$MODEL_PATH/result/annotation.txt ark,t:- | wav2vec2mdd_text/reslut/wer_per_utt_details.pl > result/ref_human_detail
align-text ark:$MODEL_PATH/result/annotation.txt  ark:$MODEL_PATH/result/hypo.txt ark,t:- | wav2vec2mdd_text/reslut/wer_per_utt_details.pl > result/human_our_detail
align-text ark:$MODEL_PATH/result/ref.txt  ark:$MODEL_PATH/result/hypo.txt ark,t:- | wav2vec2mdd_text/reslut/wer_per_utt_details.pl > result/ref_our_detail
echo $checkpoint;
python3 ../reslut/ins_del_sub_cor_analysis.py $MODEL_PATH/result 