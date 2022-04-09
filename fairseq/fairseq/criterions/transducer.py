# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
# from warprnnt_pytorch import RNNTLoss

@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    alpha_cls: float = field(
        default=0,
        metadata={"help": "coff of antiphone classification loss."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("transducer", dataclass=CtcCriterionConfig)
class TransducerCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.tgt_dict= task.target_dictionary
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.transducer_criterion = RNNTLoss(blank=self.blank_idx)

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
        self.error_calculator = None

    def forward(self, model, sample, reduce=True):
        # forward 
        encoder_out = model.encoder(tbc=False,**sample["net_input"], text=sample['text'], text_mask=sample['text_mask']) # (B, T, C)
        decoder_out = model.decoder(**sample["net_input"]) # (B, U, C)
        net_output = model.joint_network(encoder_out=encoder_out, decoder_out=decoder_out, **sample["net_input"]) # (B, T, U, C)

        # mask/lengths
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            non_padding_mask = ~net_output["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets = sample["target"]
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        # loss
        with torch.backends.cudnn.flags(enabled=False):
            rnnt_loss = self.transducer_criterion(
                F.log_softmax(net_output["net_out"],dim=-1), # (8, 112, 39, 44)
                targets.int(), # 
                input_lengths.int(),
                target_lengths.int()
            )
            lprobs = F.log_softmax(decoder_out["dec_aux_out"],dim=-1).contiguous()
            nll_loss = F.nll_loss(
                lprobs.view(-1, lprobs.size(-1)),
                targets.view(-1),
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            lprobs = F.log_softmax(encoder_out["enc_aux_out"],dim=-1).contiguous().permute(1,0,2) # btc -> tbc
            targets = sample["target"]#.masked_select(pad_mask)
            ctc_loss = F.ctc_loss(
                lprobs,
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            #loss = rnnt_loss + nll_loss + ctc_loss
            loss = rnnt_loss + ctc_loss
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),
            "rnnt_loss": utils.item(rnnt_loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            # if self.error_calculator is None:
            #     from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
            #     self.error_calculator = ErrorCalculator(
            #         decoder=model.decoder,
            #         joint_network=model.joint_network,
            #         char_list=self.tgt_dict,
            #         sym_space=self.blank_idx,
            #         sym_blank=self.blank_idx,
            #         report_cer=False,
            #         report_wer=True,
            #     )

            with torch.no_grad():
                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                # cer, wer = self.error_calculator(encoder_out["encoder_out"], sample["target"])

                    # if decoded is not None and "words" in decoded:
                    #     pred_words = decoded["words"]
                    #     w_errs += editdistance.eval(pred_words, targ_words)
                    #     wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    # else:
                    #     dist = editdistance.eval(pred_words_raw, targ_words)
                    #     w_errs += dist
                    #     wv_errs += dist

                    # w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs + 1
                logging_output["w_errors"] = w_errs + 1
                logging_output["w_total"] = w_len + 1
                logging_output["c_errors"] = c_err + 1
                logging_output["c_total"] = c_len + 1

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        rnnt_loss_sum = utils.item(sum(log.get("rnnt_loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "rnnt_loss", rnnt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
