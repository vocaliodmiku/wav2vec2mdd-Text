# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Optional, Any

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    FairseqDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )



@dataclass
class Wav2Vec2Seq2SeqConfig(Wav2Vec2AsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool  = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    joint_space_size: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    autoregressive: bool = II("task.autoregressive")


@register_model("wav2vec2_transducer", dataclass=Wav2Vec2Seq2SeqConfig)
class Wav2Vec2TModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, joint_network):
        super().__init__(encoder, decoder)
        self.joint_network = joint_network

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert cfg.autoregressive, "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg, tgt_dict)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        joint_network = cls.build_joint_network(cfg, tgt_dict)
        return Wav2Vec2TModel(encoder, decoder, joint_network)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig, tgt_dict):
        return Wav2VecEncoder(cfg, tgt_dict)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict, embed_tokens):
        return DecoderRNNT(
            dictionary=tgt_dict,
            vocab_size=len(tgt_dict), 
            embed_dim=cfg.decoder_embed_dim,
            hid_size=cfg.decoder_ffn_embed_dim,
            pad=tgt_dict.pad(),
            dropout=cfg.decoder_layerdrop,
            dropout_embed=cfg.decoder_layerdrop)

    @classmethod
    def build_joint_network(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict):
        return JointNetwork(
            vocab_size=len(tgt_dict), 
            encoder_output_size=cfg.encoder_embed_dim,
            decoder_output_size=cfg.decoder_ffn_embed_dim,
            joint_space_size=cfg.joint_space_size)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs) # (B, T, C)
        decoder_out = self.decoder(**kwargs) # (B, U, C)
        out = self.joint_network(encoder_out=encoder_out, decoder_out=decoder_out, **kwargs) # (B, T, U, C)
        return out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.proj = Linear(d, len(tgt_dict))

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)


        _x = self.proj(x)

        return {
            "enc_aux_out": _x,
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask.transpose(0, 1),  # T x B
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class DecoderRNNT(FairseqDecoder):
    def __init__(
        self,
        dictionary,
        vocab_size,
        embed_dim,
        hid_size,
        pad,
        dropout=0.0,
        dropout_embed=0.0):
        super().__init__(dictionary)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad)
        self.dropout_embed = nn.Dropout(p=dropout_embed)
        self.decoder = nn.GRU(embed_dim, hid_size, 1, batch_first=True) # TODO: batch_first???
        self.dropout = nn.Dropout(p=dropout)
        self.mlp_aux = Linear(hid_size, vocab_size)
        self.hid_size = hid_size
        self.dunits = hid_size # for espnet
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.odim = vocab_size # for espnet
        self.blank = 0 # for espnet
        self.pad = pad

    def set_device(self, device):
        self.device = device
    def set_data_type(self, data_type):
        self.data_type = data_type
    def init_state(self, batch_size):
        h = torch.zeros(batch_size, 1, self.dunits, device=self.device, dtype=self.data_type)
        return h
    def forward(self, prev_output_tokens, **kwargs):
        # _h = self.init_state(encoder_out)
        text = prev_output_tokens
        x = self.embed(text)
        x = self.dropout_embed(x)
        x, h = self.decoder(x)
        x = self.dropout(x)
        return {
            "decoder_out": x, 
            "dec_aux_out": self.mlp_aux(x)}
            
    def score(self, hyp, cache):
        vy = torch.full((1,1), hyp.yseq[-1], dtype=torch.long, device=self.device)
        str_yseq = "".join(list(map(str, hyp.yseq)))
        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)
            y, state = self.decoder(ey, hx=hyp.dec_state)
            cache[str_yseq] = (y, state)
        return y[0][0], state, vy[0]

class JointNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        encoder_output_size,
        decoder_output_size,
        joint_space_size,
        joint_activation=torch.nn.Tanh):
        super().__init__()
        self.mlp_enc = Linear(encoder_output_size, joint_space_size)
        self.mlp_dec = Linear(decoder_output_size, joint_space_size, bias=False)
        self.mlp_out = Linear(joint_space_size, vocab_size)
        self.activation = joint_activation()
       
        # self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, encoder_out, decoder_out, **kwargs):
        """
        return: (B,T,U,vocab_size)
        """
        if not isinstance(encoder_out, dict):

            return self.mlp_out( self.activation( self.mlp_enc(encoder_out) + self.mlp_dec(decoder_out))) # (B,T,U,V)
        padding_mask = encoder_out["padding_mask"]
        encoder_out = encoder_out["encoder_out"].unsqueeze(2) # (B,T,1,C)
        decoder_out = decoder_out["decoder_out"].unsqueeze(1) # (B,1,U,C)
        out = self.mlp_out( self.activation(self.mlp_enc(encoder_out) + self.mlp_dec(decoder_out) )) # (B,T,U,V)
        return {
            "net_out":out,
            "padding_mask": padding_mask,
        }


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
