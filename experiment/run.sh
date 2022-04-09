#!/disk2/plk/python/bin/
export CUDA_VISIBLE_DEVICES=0
FAIRSEQ_PATH=wav2vec2mdd_text/fairseq


MODEL=$(pwd)
python3 $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
    distributed_training.distributed_port=23333 \
    task.labels=phn \
    task.data=$MODEL/../data \
    task._name=text_condition \
    model._name=wav2vec_sigmoid \
    criterion._name=ctc_contrast \
    dataset.valid_subset=valid \
    distributed_training.distributed_world_size=1 \
    model.w2v_path=checkpoints/xlsr_53_56k.pt \
    --config-dir $MODEL/config \
    --config-name base_finetune
