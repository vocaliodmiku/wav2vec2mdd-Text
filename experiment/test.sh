#!/disk2/plk/python/bin/
# Using wav2vec2.0 to test libirpseech TEST-CLEAN data
export CUDA_VISIBLE_DEVICES=1
DATASET=/home/sait/Workplace/wav2vec/data
PROJECT=$(pwd)
# finetune 
FAIRSEQ_PATH=/home/sait/Workplace/fairseq
config_name=base_finetune # made by reffering https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/config/finetuning/base_10m.yaml
labels=phn # FIXME: labels phn doesn't work well so all related files are renamed ltr as work-around
 
# WER: 
python3 $FAIRSEQ_PATH/examples/speech_recognition/infer.py $DATASET --task text_condition \
--nbest 1 --path $PROJECT/outputs/2022-03-09/20-27-20/checkpoints/checkpoint_best.pt --gen-subset test \
--results-path $PROJECT/result --w2l-decoder viterbi \
--lm-weight 0 --word-score -1 --sil-weight 0 --criterion ctc --labels phn --max-tokens 640000 --quiet