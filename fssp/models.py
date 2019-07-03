import argparse
from typing import Dict

from fairseq import options, utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoder,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
)
from fairseq.modules import LayerNorm
import torch
import torch.nn.functional as F

from fssp.features import Feature, ASRFeature


class ASRTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of * args.encoder_layers * layers.
    Each layer is a: class: `TransformerEncoderLayer`.

    Args:
        arg: parsed command-line arguments
        embed_speech: input speech preprocessor
    """

    def __init__(self, args: argparse.Namespace, embed_speech: Feature):
        import copy
        a = copy.deepcopy(args)
        a.no_token_positional_embeddings = True
        super().__init__(a, None, embed_speech)
        self.embed_speech = embed_speech

    def forward(self, src_speech: torch.FloatTensor, src_lengths: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            src_tokens: speech in the source language of shape `(batch, src_len)`
            src_lengths: lengths of each source sentence of  `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions (B, T, C)
        x = self.embed_scale * self.embed_speech(src_speech)
        src_lengths = self.embed_speech.length(src_lengths)
        if self.embed_positions is not None:
            x += self.embed_positions(src_speech[:, :x.shape[1]])
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = torch.zeros(*x.shape[:2], dtype=torch.uint8, device=x.device)
        for i, l in enumerate(src_lengths):
            encoder_padding_mask[i, l:] = 1
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


@register_model('asr_transformer')
class ASRTransformerModel(TransformerModel):
    """
    Transformer for Automatic Speech Recognition
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--cmvn',
                            help='mean and variance of features')
        parser.add_argument('--activation-fn', default='relu',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', default=0.1,
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', default=0,
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', default=0,
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--stft-dim', type=int, metavar='N', default=400,
                            help='STFT dimension')
        parser.add_argument('--stft-stride', type=int, metavar='N', default=160,
                            help='STFT stride')
        parser.add_argument('--fbank-dim', type=int, metavar='N', default=80,
                            help='FBANK feature dimension')
        parser.add_argument('--encoder-subsample-layers', type=int, metavar='N', default=2,
                            help='encoder subsample layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', default=256,
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', default=1024,
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', default=6,
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', default=4,
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', default=None,
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', default=256,
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', default=1024,
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', default=6,
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', default=4,
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR', default=None,
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D', default=0,
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary

        encoder_embed_speech = ASRFeature(
            cmvn=args.cmvn,
            n_mels=args.fbank_dim,
            dropout=args.dropout,
            sample_rate=task.sample_rate,  # NOTE: assumes load_dataset is called before build_model
            n_fft=args.stft_dim,
            stride=args.stft_stride,
            n_subsample=args.encoder_subsample_layers,
            odim=args.encoder_embed_dim,
        )

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb
        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        setattr(encoder_embed_speech, "padding_idx", -1)  # decoder_embed_tokens.padding_idx)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        encoder = cls.build_encoder(args, encoder_embed_speech)
        return TransformerModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, embed_speech):
        return ASRTransformerEncoder(args, embed_speech)


@register_model_architecture('asr_transformer', 'asr_transformer')
def base_architecture(args):
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
