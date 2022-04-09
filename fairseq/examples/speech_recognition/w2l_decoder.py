#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flashlight decoders.
"""

import itertools as it
from logging import log
import struct
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
import viterbi

def get_data_ptr_as_bytes(tensor):
    return struct.pack("P", tensor.data_ptr())

class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc":
            self.criterion_type = "ctc"
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
            self.criterion_type = "asg"
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
        return self.decode(emissions, sample)

    #def get_emissions(self, models, encoder_input, ref=None, alt_ant=None):
    def get_emissions(self, models, encoder_input, text=None, text_mask=None, ref=None, alt_ant=None):
        """Run encoder and normalize emissions"""
        model = models[0]
        
        encoder_out = model(**encoder_input, text=text, text_mask=text_mask, ref=ref, alt_ant=alt_ant)
        if self.criterion_type == "ctc":
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out) # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
                
        elif self.criterion_type == "asg":
            emissions = encoder_out["encoder_out"]

            
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == "ctc":
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == "asg":
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions, sample):
        
        B, T, N = emissions.size()
        hypos = []
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(viterbi.get_workspace_size(B, T, N))
        viterbi.compute(
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