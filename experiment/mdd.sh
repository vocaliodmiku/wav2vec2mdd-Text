#!/disk2/plk/python/bin/
# Using wav2vec2.0 to test libirpseech TEST-CLEAN data
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
export KALDI_ROOT=/disk1/plk/kaldi

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

MODEL_PATH=$(pwd)
checkpoint=$MODEL_PATH/checkpoint_best.pt
python3 result.py /disk1/plk/wav2vec2mdd_text/data test $checkpoint || exit 1;

align-text ark:$MODEL_PATH/result/ref.txt  ark:$MODEL_PATH/result/annotation.txt ark,t:- | /disk1/plk/reslut/wer_per_utt_details.pl > result/ref_human_detail
align-text ark:$MODEL_PATH/result/annotation.txt  ark:$MODEL_PATH/result/hypo.txt ark,t:- | /disk1/plk/reslut/wer_per_utt_details.pl > result/human_our_detail
align-text ark:$MODEL_PATH/result/ref.txt  ark:$MODEL_PATH/result/hypo.txt ark,t:- | /disk1/plk/reslut/wer_per_utt_details.pl > result/ref_our_detail
echo $checkpoint;
python3 ../reslut/ins_del_sub_cor_analysis.py $MODEL_PATH/result 

# WER: 14.424838197582122   
# insert: 219
# delete: 918
# sub: 3122
# cor: 25746
# sum 29786
# sum: 30005
# 4259
# Recall: 0.5466
# Precision: 0.6403
# f1:0.5897
# TA: 0.9492 24438
# FR: 0.0508 1308
# FA: 0.4534 1931
# Correct Diag: 0.7088 1650
# Error Diag: 0.2912 678
# FAR: 0.4534
# FRR: 0.0508
# DER: 0.2912
# sub_sub 1258