#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flashlight decoders.
"""


import gc
import itertools as it
from logging import log
import os.path as osp
import pickle
import warnings
from collections import deque, namedtuple

import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from espnet.nets.beam_search_transducer import BeamSearchTransducer

try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object



class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            if "<sep>" in tgt_dict.indices:
                self.silence = tgt_dict.index("<sep>")
            elif "|" in tgt_dict.indices:
                self.silence = tgt_dict.index("|")
            else:
                self.silence = tgt_dict.eos()
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.silence = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input, sample['text'], sample['text_mask'])
        return self.decode(emissions)

    #def get_emissions(self, models, encoder_input, ref=None, alt_ant=None):
    def get_emissions(self, models, encoder_input, text=None, text_mask=None, ref=None, alt_ant=None):
        """Run encoder and normalize emissions"""
        model = models[0]
        #encoder_out = model(**encoder_input, ref=ref, alt_ant=alt_ant)
        encoder_out = model.encoder(**encoder_input, text=text, text_mask=text_mask, ref=ref, alt_ant=alt_ant)
        emissions = torch.log_softmax(encoder_out["enc_aux_out"], dim=-1)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]

class AttentionDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        
    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        model = models[0]
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        encoder_out = model.encoder(**encoder_input, text=sample['text'], text_mask=sample['text_mask'])["encoder_out"]
        B = encoder_out.size(1)
        results, ppgs = [], []
        for ind in range(B): #TODO：batch infer
            pass
        return [
            [{"tokens":results[b], "score": 0, "ppg": ppgs[b]}] for b in range(B)
        ]

class TransducerDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.beam_size = args.beam
        
    
    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        model = models[0]
        beam_search_transducer = BeamSearchTransducer(
            decoder=model.decoder,
            joint_network=model.joint_network,
            beam_size=self.beam_size,
            nbest=1,
            lm=None,
            lm_weight=0.1,
            # search_type='greedy',
            max_sym_exp=2, # number of maximum symbol expansions at each time step ("tsd")
            u_max=120, # maximum output sequence length ("alsd")
            nstep=1, # number of maximum expansion steps at each time step ("nsc")
            prefix_alpha=2, # maximum prefix length in prefix search ("nsc")
            score_norm=True, # normalize final scores by length ("default")
        )
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        encoder_out = model.encoder(**encoder_input, text=sample['text'], text_mask=sample['text_mask'])["encoder_out"]
        #encoder_out = model.get_normalized_probs([encoder_out], log_probs=True)
        B = encoder_out.size(1)
        results, ppgs = [], []
        for ind in range(B):
            nbest_hyps = beam_search_transducer(encoder_out[:,ind,:])
            debug = 1
        return [
            [{"tokens":results[b], "score": 0, "ppg": ppgs[b]}] for b in range(B)
        ]

