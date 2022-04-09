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
from collections import OrderedDict

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
        # emissions = self.get_emissions(models, encoder_input, ref=sample['ref'], alt_ant=sample['alt_ant'])
        # emissions = self.get_emissions(models, encoder_input, sample['text'], sample['text_mask'], sample['ref'], sample['alt_ant'])
        # emissions = self.get_emissions(models, encoder_input, sample['text'], sample['text_mask'])
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions, sample)

    #def get_emissions(self, models, encoder_input, ref=None, alt_ant=None):
    def get_emissions(self, models, encoder_input, text=None, text_mask=None, ref=None, alt_ant=None):
        """Run encoder and normalize emissions"""
        model = models[0]
        encoder_out = model(**encoder_input)
        # encoder_out = model(**encoder_input, ref=ref, alt_ant=alt_ant)
        #encoder_out = model(**encoder_input, text=text, text_mask=text_mask, ref=ref, alt_ant=alt_ant)
        if self.criterion_type == CriterionType.CTC:
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out) # no need to normalize emissions
            else:
                encoder_out = [encoder_out[1]]
                # emissions = model.get_normalized_probs(encoder_out, log_probs=True)
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]

            
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

    def decode(self, emissions, sample):
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
            # [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0, "text": sample["text"][b][:sample["text_lengths"][b]]}]
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        if args.lexicon:
            self.lexicon = load_words(args.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asg_transitions is None:
                N = 768
                # self.asg_transitions = torch.FloatTensor(N, N).zero_()
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )


    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos


FairseqLMState = namedtuple("FairseqLMState", ["prefix", "incremental_state", "probs"])


class FairseqLM(LM):
    def __init__(self, dictionary, model):
        LM.__init__(self)
        self.dictionary = dictionary
        self.model = model
        self.unk = self.dictionary.unk()

        self.save_incremental = False  # this currently does not work properly
        self.max_cache = 20_000

        model.cuda()
        model.eval()
        model.make_generation_fast_()

        self.states = {}
        self.stateq = deque()

    def start(self, start_with_nothing):
        state = LMState()
        prefix = torch.LongTensor([[self.dictionary.eos()]])
        incremental_state = {} if self.save_incremental else None
        with torch.no_grad():
            res = self.model(prefix.cuda(), incremental_state=incremental_state)
            probs = self.model.get_normalized_probs(res, log_probs=True, sample=None)

        if incremental_state is not None:
            incremental_state = apply_to_sample(lambda x: x.cpu(), incremental_state)
        self.states[state] = FairseqLMState(
            prefix.numpy(), incremental_state, probs[0, -1].cpu().numpy()
        )
        self.stateq.append(state)

        return state

    def score(self, state: LMState, token_index: int, no_cache: bool = False):
        """
        Evaluate language model based on the current lm state and new word
        Parameters:
        -----------
        state: current lm state
        token_index: index of the word
                     (can be lexicon index then you should store inside LM the
                      mapping between indices of lexicon and lm, or lm index of a word)

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        curr_state = self.states[state]

        def trim_cache(targ_size):
            while len(self.stateq) > targ_size:
                rem_k = self.stateq.popleft()
                rem_st = self.states[rem_k]
                rem_st = FairseqLMState(rem_st.prefix, None, None)
                self.states[rem_k] = rem_st

        if curr_state.probs is None:
            new_incremental_state = (
                curr_state.incremental_state.copy()
                if curr_state.incremental_state is not None
                else None
            )
            with torch.no_grad():
                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cuda(), new_incremental_state
                    )
                elif self.save_incremental:
                    new_incremental_state = {}

                res = self.model(
                    torch.from_numpy(curr_state.prefix).cuda(),
                    incremental_state=new_incremental_state,
                )
                probs = self.model.get_normalized_probs(
                    res, log_probs=True, sample=None
                )

                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cpu(), new_incremental_state
                    )

                curr_state = FairseqLMState(
                    curr_state.prefix, new_incremental_state, probs[0, -1].cpu().numpy()
                )

            if not no_cache:
                self.states[state] = curr_state
                self.stateq.append(state)

        score = curr_state.probs[token_index].item()

        trim_cache(self.max_cache)

        outstate = state.child(token_index)
        if outstate not in self.states and not no_cache:
            prefix = np.concatenate(
                [curr_state.prefix, torch.LongTensor([[token_index]])], -1
            )
            incr_state = curr_state.incremental_state

            self.states[outstate] = FairseqLMState(prefix, incr_state, None)

        if token_index == self.unk:
            score = float("-inf")

        return outstate, score

    def finish(self, state: LMState):
        """
        Evaluate eos for language model based on the current lm state

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        return self.score(state, self.dictionary.eos())

    def empty_cache(self):
        self.states = {}
        self.stateq = deque()
        gc.collect()


class W2lFairseqLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        self.lexicon = load_words(args.lexicon) if args.lexicon else None
        self.idx_to_wrd = {}

        checkpoint = torch.load(args.kenlm_model, map_location="cpu")

        if "cfg" in checkpoint and checkpoint["cfg"] is not None:
            lm_args = checkpoint["cfg"]
        else:
            lm_args = convert_namespace_to_omegaconf(checkpoint["args"])

        with open_dict(lm_args.task):
            lm_args.task.data = osp.dirname(args.kenlm_model)

        task = tasks.setup_task(lm_args.task)
        model = task.build_model(lm_args.model)
        model.load_state_dict(checkpoint["model"], strict=False)

        self.trie = Trie(self.vocab_size, self.silence)

        self.word_dict = task.dictionary
        self.unk_word = self.word_dict.unk()
        self.lm = FairseqLM(self.word_dict, model)

        if self.lexicon:
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                if self.unit_lm:
                    word_idx = i
                    self.idx_to_wrd[i] = word
                    score = 0
                else:
                    word_idx = self.word_dict.index(word)
                    _, score = self.lm.score(start_state, word_idx, no_cache=True)

                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                [],
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []

        def idx_to_word(idx):
            if self.unit_lm:
                return self.idx_to_wrd[idx]
            else:
                return self.word_dict[idx]

        def make_hypo(result):
            hypo = {"tokens": self.get_tokens(result.tokens), "score": result.score}
            if self.lexicon:
                hypo["words"] = [idx_to_word(x) for x in result.words if x >= 0]
            return hypo

        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append([make_hypo(result) for result in nbest_results])
            self.lm.empty_cache()

        return hypos

class TeacherCTCDecoder(object):
    def __init__(self, args, tgt_dict) -> None:
        super().__init__()
        self.tgt_dict = tgt_dict
        self.blank = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.alignment = self.parse_align("/home/sait/Workplace/wav2vec/experiments/tran_enc_sigmoid_xlsr/result/ref_our_detail") # hijack 
        self.scorer = CtcScorer(0, self.tgt_dict.pad(), self.tgt_dict)
        debug = 1

    @staticmethod
    def parse_align(txt):
        def func(line):
            line = line.strip('\n').split(' ')
            info = line[2:]
            return [i for i in info if i]

        data = {}
        lines = open(txt, 'r').readlines()
        idx = 0
        while idx < len(lines):
            ref = func(lines[idx])
            idx+=1
            ann = func(lines[idx])
            idx+=1
            ops = func(lines[idx])
            idx+=1
            inf = func(lines[idx])
            id = lines[idx].strip('\n').split(' ')[0]
            idx+=1
            data[id] = {
                "ref": ref, "ann": ann, "ops": ops
            }
        return data
    
    def generate(self, models, sample, **unused):
        encoder_input = {
            k: v for k, v in sample["net_input"].items()
        }
        model = models[0]
        encoder_out = model(**encoder_input, text=sample['text'], text_mask=sample['text_mask'])
        attn = encoder_out["encoder_attn"]
        encoder_out = model.get_normalized_probs(encoder_out, log_probs=True).cpu()
        B = encoder_out.size(1)
        results, ppgs = [], []
        for ind in range(B):
            id = sample["iid"][ind].split('.')[0].split('/')
            id = id[1] + '%' + id[-1]
            if id != f"NJS%arctic_a0112":
                results.append([{"tokens":[0]}])
                continue
            with open("gate_attn.pkl", 'wb') as f:
                att = attn[ind]
                pickle.dump(att,f)
            align = self.alignment[id]
            ref, ann = align['ref'], align['ann']
            result = self.ctc_score_TF(encoder_out[:,ind,:], ref, ann) # teacher force decode
            results.append(result)
        return results

    def ctc_score_TF(self, pp, ref, ann):
        res, ppg  = self.scorer(pp, ref, ann, len(ref))
        ppg = np.array(ppg)
        with open("ppg_wo.pkl", 'wb') as f:
            pickle.dump(ppg,f)
        return 
        


class TeacherAttnDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.blank =tgt_dict.bos()
        self.eos=tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
    
    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items()
        }
        model = models[0]
        encoder_out, attention = model(**encoder_input, text=sample['text'], text_mask=sample['text_mask'])
        encoder_out = model.get_normalized_probs([encoder_out], log_probs=True)
        B = encoder_out.size(0)
        results, ppgs = [], []
        for ind in range(B):
            result = self.att_score_TF(encoder_out[ind,:,:], sample["text"][ind], sample["text_lengths"][ind]) # teacher force decode
            results.append(result)

        return [
            [{"tokens":results[b], "score": 0, "ppg": None, "text": sample["text"][b][:sample["text_lengths"][b]]}] for b in range(B)
        ]

    def att_score_TF(self, ppg, text, length):
        result = []
        for ind in range(length):
            pred = torch.argmax(ppg[ind]).cpu().item()
            result.append(pred)
        return result

class TeacherForceDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.ctc_scorer = CtcScorer(blank=tgt_dict.bos(), eos=tgt_dict.eos(), vocab_size=len(tgt_dict))
    
    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        model = models[0]
        encoder_out = model(**encoder_input, text=sample['text'], text_mask=sample['text_mask'])
        encoder_out = model.get_normalized_probs(encoder_out, log_probs=True)
        B = encoder_out.size(1)
        results, ppgs = [], []
        for ind in range(B):
            result, ppg = self.ctc_score_TF(encoder_out[:,ind,:], sample["text"][ind], sample["text_lengths"][ind]) # teacher force decode
            results.append(result)
            ppgs.append(ppg)

        return [
            [{"tokens":results[b], "score": 0, "ppg": ppgs[b]}] for b in range(B)
        ]


    def ctc_score_TF(self, encoder_out, target, target_length):
        return self.ctc_scorer(encoder_out, target, target_length)


class CtcScorer(object):
    def __init__(self, blank, eos, tgt_dict):
        self.blank = blank
        self.eos = eos
        self.encoder_output = None
        self.time_step = None
        self.logzero = -100000000000.0
        self.vocab_size = len(tgt_dict)
        self.tgt_dict = tgt_dict

    def init_state(self, encoder_output):
        self.encoder_output = encoder_output.numpy()
        self.time_step = encoder_output.size(0)
        dp = encoder_output.new_ones(self.time_step, 2) * self.logzero
        dp[0,1] = encoder_output[0, self.blank]
        for i in range(1, self.time_step):
            dp[i, 1] = dp[i - 1, 1] + self.encoder_output[i, self.blank] # note here is logprob 
        return dp.numpy() # state

    def __call__(self, encoder_out, target, prediction, target_length):
        prev_dp= self.init_state(encoder_out)
        prefix, result, ppgs, track = [self.eos,], [], [], []
        for ind in range(target_length):
            log_psi, cur_dp = self.score(prefix, prev_dp)
            if prediction[ind] == "<eps>": # del
                continue
            cur_targ = self.tgt_dict.indices[prediction[ind]]
            cur_pred = np.argmax(log_psi)
            cur_tgt_, cur_pred_ = target[ind], prediction[ind]
            prefix.append(cur_targ)
            prev_dp = cur_dp[cur_targ] #  Teacher force
            if prediction[ind] != "<eps>": # insert error
                # ppgs.append(self.build_ppgdict(cur_tgt_, log_psi))
                ppg = np.logaddexp(prev_dp[:,0], prev_dp[:,1])
                ppgs.append(ppg)
                
        return torch.Tensor(result), ppgs
    
    def build_ppgdict(self, cur_tgt, log_psi):
        data = {}
        for i, s in enumerate(self.tgt_dict.symbols):
            data[s] = log_psi[i]
        data = sorted(data.items(), key=lambda x: -x[1])
        return data



    def score(self, prefix, prev_state):
        output_length = len(prefix) - 1
        dp = np.ndarray((self.time_step, 2, self.vocab_size), dtype=np.float32)
        if output_length == 0: # sos
            dp[0,0] = self.encoder_output[0] # assign the logprob of encoder to the dp as init state
            dp[0,1] = self.logzero # prob 1
        else:
            dp[output_length - 1] = self.logzero
        
        prev_sum = np.logaddexp(prev_state[:,0], prev_state[:,1])
        last = prefix[-1]
        if output_length > 0:
            log_phi = np.ndarray((self.time_step, self.vocab_size), dtype=np.float32)
            for i in range(self.vocab_size):
                log_phi[:,i] = prev_sum if i != last else prev_state[:, 1]
        else:
            log_phi = prev_sum
        start = max(output_length, 1)
        log_psi = dp[start - 1, 0] # end w/o blank

        for t in range(start, self.time_step):
            dp[t, 0] = np.logaddexp( dp[t-1, 0], log_phi[t-1]) + self.encoder_output[t] # (transition directly from prev token + w/o transition with ok dp) * prob of current time 
            dp[t, 1] = np.logaddexp( dp[t-1, 0], dp[t-1, 1]) + self.encoder_output[t, self.blank] # (transition from ok dp[0] + transition from ok dp[1]) * cur prob of blank
            log_psi = np.logaddexp( log_psi, log_phi[t-1] + self.encoder_output[t])
        
        log_psi[self.eos] = prev_sum[-1]
        log_psi[self.blank] = self.logzero
        return log_psi, np.rollaxis(dp, 2)


    # def _score(self, postfix, next_state):

