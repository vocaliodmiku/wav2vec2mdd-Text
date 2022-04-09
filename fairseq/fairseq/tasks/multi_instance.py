# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset, encoders, BaseWrapperDataset, data_utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


class MultiInstanceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        texts,
        pad,
        eos,
        batch_targets,
        process_label=None,
        add_to_input=False,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.texts = texts
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.labels[index])
        )
    
    def get_text(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.texts[index])
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        item["text"] = self.get_text(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        text = [s["text"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["text_lengths"] = torch.LongTensor([len(t) for t in text])
            text = data_utils.collate_tokens(text, pad_idx=self.pad, left_pad=False).long()
            collated["ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target

        lens = collated["text_lengths"]
        padding_mask = torch.BoolTensor(text.shape).fill_(False)
        text_size = text.shape[1]
        for i, size in enumerate(lens):
            diff = text_size - size
            assert diff >=0 
            if diff == 0:
                continue
            else:
                padding_mask[i, -diff:] = True
        collated["text"] = text
        collated["text_mask"] = padding_mask

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1).long()
            collated["net_input"]["prev_output_tokens"] = torch.cat(
                [eos, target], dim=-1
            ).long()
            collated["ntokens"] += target.size(0)
        return collated


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "number of buckets"
        },
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")

    tpu: bool = II("common.tpu")


@register_task("multi_instance", dataclass=AudioPretrainingConfig)
class MultiInstanceTask(FairseqTask):
    """"""

    cfg: AudioPretrainingConfig

    def __init__(
        self,
        cfg: AudioPretrainingConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(
                self.cfg.data, f"dict.{self.cfg.labels}.txt"
            )
            return Dictionary.load(dict_path)
        return None

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            args = [
                'mask_length',
                'mask_prob',
                'mask_selection',
                'mask_other',
                'no_mask_overlap',
                'mask_min_space',
                'mask_channel_length',
                'mask_channel_prob',
                'mask_channel_selection',
                'mask_channel_other',
                'no_mask_channel_overlap',
                'mask_channel_min_space',
                'encoder_embed_dim',
                'conv_feature_layers',
            ]
            return {arg: cfg[arg] for arg in args}
        else:
            return {}

    def load_dataset(
            self, split: str, task_cfg: FairseqDataclass = None, **kwargs
    ):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=task_cfg.get('sample_rate', self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            compute_mask_indices=(
                self.cfg.precompute_mask_indices or self.cfg.tpu
            ),
            **self._get_mask_precompute_kwargs(task_cfg),
        )

        if self.cfg.tpu and task_cfg['mask_channel_prob'] == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            text_path = os.path.join(data_path, f"{split}.text")
            with open(label_path, "r") as f:
                labels = [
                    line for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]
            with open(text_path, "r") as f:
                texts = [
                    line for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]

            assert len(texts) == len(labels) == len(self.datasets[split]), (
                    f"labels(or texts) length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            process_label = LabelEncoder(self.target_dictionary)
        
            self.datasets[split] = MultiInstanceDataset(
                self.datasets[split],
                labels,
                texts,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.get('autoregressive', False),
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        
        model = super().build_model(model_cfg)
        print(model)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
