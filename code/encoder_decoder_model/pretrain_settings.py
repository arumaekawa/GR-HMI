import argparse
import datetime
import logging
import os
import random

import GPUtil
import numpy as np
import torch

from .models import (
    BertConfig,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

logger = logging.getLogger(__name__)


FILL_VAL = -1
LEN_FACTOR = 1.163
MEMORY_FACTOR = 0.18
TURING_ARCHS = {"Tesla V100", "2080 Ti"}
ENCODER_CLASSES = {
    "bert-base-uncased": (BertModel, BertTokenizer, BertConfig),
    # 'roberta-base': (RobertaModel, RobertaTokenizer, RobertaConfig),
}
DECODER_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
    # 'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig)
}
ENCODER_SAVE_NAME = "encoder-"
DECODER_SAVE_NAME = "decoder-"
FINAL_ENCODER_SAVE_NAME = "encoder-finish"
FINAL_DECODER_SAVE_NAME = "decoder-finish"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1000)
    # environment
    parser.add_argument("--pretrain_data_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir_root", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--fp32", action="store_true")
    # methods
    parser.add_argument(
        "--encoder_model_name",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased"],
    )
    parser.add_argument(
        "--decoder_model_name", type=str, default="gpt2", choices=["gpt2"]
    )
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--with_lm", action="store_true")
    parser.add_argument("--only_lm", action="store_true")
    # data
    parser.add_argument(
        "--train_data_path", type=str, default="wiki40b/tokenized_train_data.pkl"
    )
    parser.add_argument("--valid_data_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--encoder_max_len", type=int, default=0)
    parser.add_argument("--decoder_max_len", type=int, default=0)
    parser.add_argument("--min_seq_len", type=int, default=10)
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--sample_top_p", type=float, default=0.0)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    # learning
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--min_batch_size", type=int, default=0)
    parser.add_argument("--min_n_steps", type=int, default=50000)
    args = parser.parse_args()

    # set random seed
    set_random_seed(args.seed)

    if args.debug:
        args.logging_steps = 1
        # torch.manual_seed(0)
        set_random_seed(0)
        # torch.backends.cudnn.deterministric = True

    # model name
    if not args.only_lm:
        args.model_name = "{}_{}_{}".format(
            args.encoder_model_name, args.decoder_model_name, args.feature_dim
        )
        if args.with_lm:
            args.model_name = args.model_name + "_with_lm"
    else:
        args.model_name = args.decoder_model_name + "_only_lm"
        args.with_lm = True

    # save directory path
    args.pretrained_model_dir = os.path.join(
        args.pretrained_model_dir_root, args.model_name
    )

    # set gpu devices
    args.device_ids = GPUtil.getAvailable(
        maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus
    )
    if len(args.device_ids) == 0:
        logger.error("No available GPUs!")
        raise NotImplementedError("No CPU mode available!")

    if len(args.device_ids) < args.n_gpus:
        logger.warning(
            "Available number of GPU = {} < n_gpus = {}".format(
                len(args.device_ids), args.n_gpus
            )
        )
        args.n_gpus = len(args.device_ids)
        logger.warning("Continue training with {} GPUs".format(args.n_gpus))

    args.device_ids = list(range(len(args.device_ids)))
    torch.cuda.set_device(0)

    # memory size of each gpu device
    gpus = GPUtil.getGPUs()
    gpu_names = [gpus[device_id].name for device_id in args.device_ids]
    if not all(
        any(turing_arch in gpu_name for turing_arch in TURING_ARCHS)
        for gpu_name in gpu_names
    ):
        logger.warning("Not all gpus support fp16 training! Will use fp32 instead.")
        args.fp32 = True
    if args.decoder_model_name == "openai-gpt":
        args.fp32 = True  # openai-gpt currently doesn't support fp16
    if not args.fp32:
        global MEMORY_FACTOR
        MEMORY_FACTOR = MEMORY_FACTOR * 1.4
    args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]
    args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus - 1))
    for i in range(1, args.n_gpus):
        args.memory_sizes[i] = args.memory_sizes[i] * 1.04
    # set max batch size according to gpu memory size
    if args.train_batch_size <= 0:
        args.train_batch_size = [
            int(memory_size * MEMORY_FACTOR) for memory_size in args.memory_sizes
        ]
    if args.test_batch_size <= 0:
        args.test_batch_size = [
            int(memory_size * MEMORY_FACTOR) for memory_size in args.memory_sizes
        ]

    # special tokens
    encoder_special_tokens = {
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "ans_token": "[ANS]",
    }
    decoder_special_tokens = {
        "ans_token": "__ans__",
        "pad_token": "__pad__",
        "unk_token": "__unk__",
        "eos_token": "<|endoftext|>",
        "gen_token": "__gen__",
    }

    # model and tokenizer classes for encoder and decoder
    encoder_class, encoder_tokenizer_class, encoder_config_class = ENCODER_CLASSES[
        args.encoder_model_name
    ]
    decoder_class, decoder_tokenizer_class, decoder_config_class = DECODER_CLASSES[
        args.decoder_model_name
    ]
    encoder_tokenizer = encoder_tokenizer_class.from_pretrained(args.encoder_model_name)
    decoder_tokenizer = decoder_tokenizer_class.from_pretrained(args.decoder_model_name)

    # load config of pretrained model
    encoder_config = encoder_config_class.from_pretrained(args.encoder_model_name)
    decoder_config = decoder_config_class.from_pretrained(args.decoder_model_name)

    # set max length
    if args.encoder_max_len == 0:
        args.encoder_max_len = encoder_config.max_position_embeddings
    if args.decoder_max_len == 0:
        args.decoder_max_len = decoder_config.n_positions

    return (
        args,
        encoder_config,
        encoder_class,
        encoder_tokenizer,
        encoder_config_class,
        decoder_config,
        decoder_class,
        decoder_tokenizer,
        decoder_config_class,
        encoder_special_tokens,
        decoder_special_tokens,
    )


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated / 1000 - last / 1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated // 1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = (
        "%(asctime)s - %(uptime)s - %(relative)ss -"
        " %(levelname)s - %(name)s - %(message)s"
    )
    logging.basicConfig(
        format=logging_format, filename=filename, filemode="a", level=logging.INFO
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())


(
    args,
    ENCODER_CONFIG,
    ENCODER_CLASS,
    ENCODER_TOKENIZER,
    ENCODER_CONFIG_CLASS,
    DECODER_CONFIG,
    DECODER_CLASS,
    DECODER_TOKENIZER,
    DECODER_CONFIG_CLASS,
    ENCODER_SPECIAL_TOKENS,
    DECODER_SPECIAL_TOKENS,
) = parse_args()
