from fairseq import options, utils
from fairseq.tasks import FairseqTask, register_task

from fssp.data import ASRDataset


@register_task('asr')
class ASRTask(FairseqTask):
    """Transcribe from (source) speech to (target) text

    Args:
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target text

    .. note::
        The ASR task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--wav', nargs='+',
                            help='speech wav.scp list e.g., ./data/train/wav.scp ./data/dev/wav.scp ./data/test/text')
        parser.add_argument('--text', nargs='+',
                            help='train text list e.g., ./data/train/text ./data/dev/text ./data/test/text')
        parser.add_argument('--dict',
                            help='dict txt path')
        parser.add_argument('--max-source-positions', default=9999999999, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--sample-rate', default=0, type=int, metavar='N',
                            help='sample rate of wav (default: 0 for auto)')
        # fmt: on

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.sample_rate = args.sample_rate

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        tgt_dict = cls.load_dictionary(args.dict)
        print('|  dictionary: {} types'.format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        wav = None
        for path in self.args.wav:
            if split in path.split("/"):
                wav = path
        if wav is None:
            raise ValueError(f"unknown wav.scp split: {split}")
        text = None
        for path in self.args.text:
            if split in path.split("/"):
                text = path
        if text is None:
            raise ValueError(f"unknown text split: {split}")

        self.datasets[split] = ASRDataset(
            wav,
            text,
            self.tgt_dict,
            self.args.sample_rate,
            # max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )
        self.sample_rate = self.datasets[split].sample_rate

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        raise NotImplementedError()
        # return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
