import torch

from fairseq.modules import LayerNorm


class Feature(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def embedding_dim(self):
        raise NotImplementedError

    def length(self, src_lengths: torch.LongTensor) -> torch.LongTensor:
        """Calculate output length

        Args
            src_lengths: lengths of source tensor

        Returns:
            processed lengths
        """
        return src_lengths


class STFT(Feature):
    """Short Time Fourie Transformation from Wave Signal

    Note: default values are retrieved from kaldi
    """

    def __init__(self, n_fft: int = 400, stride: int = 160):
        super().__init__()
        self.n_fft = n_fft
        self.stride = stride

    @property
    def embedding_dim(self):
        return self.n_fft // 2 + 1

    def length(self, src_lengths: torch.LongTensor) -> torch.LongTensor:
        return src_lengths // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply STFT

        Args:
            x: input wave signal to the layer of shape `(batch, seq_len)`

        Returns:
            STFT output of shape `(batch, seq_len // stride + 1, n_fft // 2 + 1, 2)`,
            where the last dim has (real, imag)
        """
        assert x.dim() == 2
        return torch.stft(x, self.n_fft, self.stride).permute(0, 2, 1, 3)


class Power(Feature):
    """Power computation on STFT feature"""

    def __init__(self, mode: str = "kaldi"):
        super().__init__()
        # TODO(karita): implement more?
        assert mode in ["kaldi"]
        self.mode = "kaldi"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate power

        Args:
            x: STFT output of shape `(batch, seq_len, idim, 2)`, where the last dim has (real, imag)

        Returns:
            Power of STFT feature of shape `(batch, seq_len, idim)`
        """
        assert x.dim() == 4
        if self.mode == "kaldi":
            return x[:, :, :, 0] ** 2 + x[:, :, :, 1] ** 2
        else:
            raise NotImplementedError(self.mode)


class LogMelFilterbank(Feature):
    """Log Mel Filterbank from STFT"""

    def __init__(self, idim: int, odim: int = 80, sample_rate: int = 16000):
        import librosa
        super().__init__()
        melmat = librosa.filters.mel(sr=16000, n_fft=idim, n_mels=odim, norm=1, htk=False)
        self.register_buffer('melmat', torch.from_numpy(melmat.T).float())

    @property
    def embedding_dim(self):
        return self.melmat.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram into log mel filterbank

        Args:
            x(Tensor): input STFT feature to the layer of shape `(batch, seq_len, idim)`

        Returns:
            fbank output of shape `(batch, seq_len, odim)`
        """
        y = x.matmul(self.melmat)
        return y


class Conv1dSubsampling(Feature):
    """Convolutional Subsampling"""

    def __init__(self, idim, odim, dropout, kernel_size=3, stride=2, n_layers=2):
        super().__init__()
        self.dropout = dropout
        self.odim = odim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(n_layers):
            d = idim if i == 0 else odim
            conv = torch.nn.Conv1d(d, odim, kernel_size=kernel_size, stride=stride)
            self.convs.append(conv)
            self.norms.append(LayerNorm(odim))

    @property
    def embedding_dim(self):
        return self.odim

    def length(self, src_lengths):
        import math
        for c in self.convs:
            src_lengths = ((src_lengths - 2 * c.padding[0] - c.dilation[0]
                            * (c.kernel_size[0] - 1) - 1) / c.stride[0] + 1).long()
        return src_lengths

    def forward(self, x):
        """Subsample feature with learnable convolutions

        Args:
            x(Tensor): input to the layer of shape `(batch, seq_len, idim)`

        Returns:
            encoded output of shape `(batch, subsampled seq_len, odim)`
        """
        if len(self.convs) == 0:
            return x
        x = x.contiguous()
        for c, n in zip(self.convs, self.norms):
            x = c(x.permute(0, 2, 1))  # (B, C, T)
            x = n(x.permute(0, 2, 1))  # (B, T, C)
            d = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = torch.relu(d)
        return d


class CMVN(Feature):
    def __init__(self, mean, stddev):
        super().__init__()
        assert mean.dim() == 1
        assert mean.shape == stddev.shape
        self.register_buffer('mean', mean.view(1, 1, -1))
        self.register_buffer('stddev', stddev.view(1, 1, -1))

    def forward(self, x):
        return (x - self.mean) / self.stddev


class ASRFeature(Feature):
    """ASR Feature Extractor"""

    def __init__(self, odim, dropout=0.0, n_mels=80, sample_rate=16000, n_fft=400, stride=160, n_subsample=2, cmvn=None):
        super().__init__()
        self.odim = odim
        net = [
            STFT(n_fft, stride),
            Power(),
            LogMelFilterbank(n_fft, n_mels, sample_rate),
        ]
        if cmvn is not None:
            if isinstance(cmvn, str):
                d = torch.load(cmvn)
                net.append(CMVN(d["mean"], d["stddev"]))
            else:
                raise ValueError(f"invalid cmvn {cmvn}")
        net.append(Conv1dSubsampling(n_mels, odim, dropout, n_layers=n_subsample))
        self.net = torch.nn.Sequential(*net)

    @property
    def embedding_dim(self):
        return self.odim

    def length(self, src_lengths: torch.LongTensor) -> torch.LongTensor:
        for n in self.net:
            src_lengths = n.length(src_lengths)
        return src_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
