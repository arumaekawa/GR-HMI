import argparse
import datetime
import json
import logging
import os
import random

import GPUtil
import numpy as np
import torch
from encoder_decoder_model.models import (
    CONFIG_NAME,
    BertConfig,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from hippocampus_module import HIPPOCAMPUS_MODULE_CLASSES

logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1
LEN_FACTOR = 1.163
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "hmi-lamol": 0.35,
    "lamol": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
}
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
LLL_METHODS = ["hmi-lamol", "lamol"]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--comment", type=str, default="")
    # environment
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir_root", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--pretrain_with_lm", action="store_true")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--fp32", action="store_true")
    # methods
    parser.add_argument(
        "--seq_train_type",
        type=str,
        default="hmi-lamol",
        choices=["hmi-lamol", "lamol", "finetune", "multitask", "mas", "ewc", "gem"],
    )
    parser.add_argument(
        "--encoder_model_name",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased"],
    )
    parser.add_argument(
        "--decoder_model_name", type=str, default="gpt2", choices=["gpt2", "openai-gpt"]
    )
    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--use_sep", action="store_true")
    # tasks
    parser.add_argument("--tasks", nargs="+", default=["squad2"])
    parser.add_argument("--skip_tasks", nargs="+")
    parser.add_argument("--unbound", type=int, default=0)
    # learning
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    #   lr
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    #   batch size
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    #   number of training steps
    parser.add_argument("--n_train_epochs", type=int, default=9)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--dynamic_epochs", action="store_true")
    #   others
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--tokens_weight", type=float, default=5)
    # decoding
    parser.add_argument("--top_k_qa", type=int, default=1)
    parser.add_argument("--top_p_qa", type=float, default=0.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    # for `hmi-lamol` and `lamol`
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--gen_lambda", type=float, default=0.25)
    # for `hmi-lamol`
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument(
        "--hc_module_type",
        type=str,
        default="random",
        choices=list(HIPPOCAMPUS_MODULE_CLASSES.keys()),
    )
    parser.add_argument("--max_memory_size", type=int, default=None)
    # for `lamol`
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.0)
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    # for `gem`
    parser.add_argument("--qp_margin", type=float, default=0.5)
    # for `ewc` and `mas`
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    args = parser.parse_args()

    # set random seed
    set_random_seed(args.seed)

    if args.debug:
        args.logging_steps = 1
        # torch.manual_seed(0)
        set_random_seed(0)
        # torch.backends.cudnn.deterministric = True  # <- included above

    # model name
    if args.seq_train_type == "hmi-lamol":
        args.model_name = "_".join(
            (args.encoder_model_name, args.decoder_model_name, str(args.feature_dim))
        )
        if args.pretrain_with_lm:
            args.model_name += "_with_lm"
    else:
        args.model_name = args.decoder_model_name
        if args.pretrained_model_dir:
            args.model_name += "_pretrained"
    # save directory name
    model_dir_name_list = ["_".join(args.tasks)]
    if args.real_sample:
        model_dir_name_list.append("real")
    elif args.add_task_tokens:
        model_dir_name_list.append("task")
    else:
        model_dir_name_list.append("gen")
    if args.seq_train_type in LLL_METHODS:
        model_dir_name_list.append(str(args.gen_lm_sample_percentage))
    if args.comment:
        model_dir_name_list.append(args.comment)
    model_dir_name_list.append(f"seed_{args.seed}")
    # save directory path
    args.model_dir_root = os.path.join(
        args.model_dir_root,
        args.seq_train_type,
        args.model_name,
        "_".join(model_dir_name_list),
    )

    # gpu device settings
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

    # memory size of each devices
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
        MEMORY_FACTOR = dict([k, v * 1.4] for k, v in MEMORY_FACTOR.items())
    args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]
    args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus - 1))
    for i in range(1, args.n_gpus):
        args.memory_sizes[i] = args.memory_sizes[i] * 1.04
    # Set batch size according to memory size
    if args.train_batch_size <= 0:
        args.train_batch_size = [
            int(memory_size * MEMORY_FACTOR[args.seq_train_type])
            for memory_size in args.memory_sizes
        ]
    if args.test_batch_size <= 0:
        args.test_batch_size = [
            int(memory_size * MEMORY_FACTOR[args.seq_train_type])
            for memory_size in args.memory_sizes
        ]

    # flag for encoder settings
    use_encoder = args.seq_train_type == "hmi-lamol"

    # special tokens
    encoder_special_tokens = (
        {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "ans_token": "[ANS]",
        }
        if use_encoder
        else None
    )
    decoder_special_tokens = {
        "ans_token": "__ans__",
        "pad_token": "__pad__",
        "unk_token": "__unk__",
        "eos_token": "<|endoftext|>",
    }
    if args.use_sep:
        if use_encoder:
            encoder_special_tokens["sep_token"] = "[SEP]"
        decoder_special_tokens["sep_token"] = "__sep__"

    # classes of model, tokenizer and config
    encoder_class, encoder_tokenizer_class, encoder_config_class = (
        ENCODER_CLASSES[args.encoder_model_name] if use_encoder else (None, None, None)
    )
    decoder_class, decoder_tokenizer_class, decoder_config_class = DECODER_CLASSES[
        args.decoder_model_name
    ]

    # load pretrained tokenizers
    if args.pretrained_model_dir is not None:
        encoder_tokenizer = (
            encoder_tokenizer_class.from_pretrained(
                os.path.join(args.pretrained_model_dir, "encoder")
            )
            if use_encoder
            else None
        )
        decoder_tokenizer = decoder_tokenizer_class.from_pretrained(
            os.path.join(args.pretrained_model_dir, "decoder")
        )
    else:
        encoder_tokenizer = (
            encoder_tokenizer_class.from_pretrained(args.encoder_model_name)
            if use_encoder
            else None
        )
        decoder_tokenizer = decoder_tokenizer_class.from_pretrained(
            args.decoder_model_name
        )
    # add special tokens to tokenizers
    if use_encoder:
        encoder_tokenizer.add_tokens(list(encoder_special_tokens.values()))
    decoder_tokenizer.add_tokens(list(decoder_special_tokens.values()))
    # special token ids dictionary
    encoder_special_token_ids = (
        {
            k: encoder_tokenizer.convert_tokens_to_ids(v)
            for k, v in encoder_special_tokens.items()
        }
        if use_encoder
        else None
    )
    decoder_special_token_ids = {
        k: decoder_tokenizer.convert_tokens_to_ids(v)
        for k, v in decoder_special_tokens.items()
    }

    # load config of pretrained language model
    if args.pretrained_model_dir is not None:
        encoder_config = (
            encoder_config_class.from_json_file(
                os.path.join(args.pretrained_model_dir, "encoder", CONFIG_NAME)
            )
            if use_encoder
            else None
        )
        decoder_config = decoder_config_class.from_json_file(
            os.path.join(args.pretrained_model_dir, "decoder", CONFIG_NAME)
        )
    else:
        encoder_config = (
            encoder_config_class.from_pretrained(args.encoder_model_name)
            if use_encoder
            else None
        )
        decoder_config = decoder_config_class.from_pretrained(args.decoder_model_name)

    # tokens weight
    tokens_weight = torch.ones([len(decoder_tokenizer)], dtype=torch.float).cuda()
    tokens_weight[decoder_special_token_ids["ans_token"]] = args.tokens_weight
    if args.use_sep:
        tokens_weight[decoder_special_token_ids["sep_token"]] = args.tokens_weight

    # max length
    args.encoder_max_len = (
        encoder_config.max_position_embeddings if use_encoder else None
    )
    args.decoder_max_len = decoder_config.n_positions

    # hippocampus module (for hmi-lamol)
    if args.seq_train_type == "hmi-lamol":
        hippocampus_module_class = HIPPOCAMPUS_MODULE_CLASSES[args.hc_module_type]
    else:
        hippocampus_module_class = None

    # data attributions file
    data_attrs_path = os.path.join(BASE_DIR, "data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)

    # set training epochs
    if args.seq_train_type == "multitask":
        args.n_train_epochs = {"_".join(args.tasks): args.n_train_epochs}
    elif args.unbound:
        pass
    else:
        if "gem" in args.seq_train_type:
            args.memory_data = []
        if args.dynamic_epochs:
            data_sizes = {
                task: data_attrs[task]["train"]["data_size"] for task in args.tasks
            }
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {
                d[0]: min(args.max_n_epochs, max_total_data_size // d[1])
                for d in data_sizes.items()
            }
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}

    return (
        args,
        encoder_config,
        decoder_config,
        encoder_class,
        decoder_class,
        encoder_tokenizer,
        decoder_tokenizer,
        encoder_config_class,
        decoder_config_class,
        encoder_special_token_ids,
        decoder_special_token_ids,
        encoder_special_tokens,
        decoder_special_tokens,
        hippocampus_module_class,
        data_attrs,
        tokens_weight,
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
        "%(asctime)s - %(uptime)s - %(relative)ss - "
        "%(levelname)s - %(name)s - %(message)s"
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
    DECODER_CONFIG,
    ENCODER_CLASS,
    DECODER_CLASS,
    ENCODER_TOKENIZER,
    DECODER_TOKENIZER,
    ENCODER_CONFIG_CLASS,
    DECODER_CONFIG_CLASS,
    ENCODER_SPECIAL_TOKEN_IDS,
    DECODER_SPECIAL_TOKEN_IDS,
    ENCODER_SPECIAL_TOKENS,
    DECODER_SPECIAL_TOKENS,
    HIPPOCAMPUS_MODULE_CLASS,
    DATA_ATTRS,
    TOKENS_WEIGHT,
) = parse_args()


TASK_DICT = {
    "squad1": {
        "train": os.path.join(args.data_dir, "squad-train-v1.1.json"),
        "eval": os.path.join(args.data_dir, "squad-dev-v1.1.json"),
        "test": os.path.join(args.data_dir, "squad-dev-v1.1.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "squad2": {
        "train": os.path.join(args.data_dir, "squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "squad-dev-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "iwslt.en.de": {
        "train": os.path.join(args.data_dir, "iwslt.en.de_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "iwslt.en.de_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "iwslt.en.de_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "cnn_dailymail": {
        "train": os.path.join(args.data_dir, "cnn_dailymail_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "cnn_dailymail_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "cnn_dailymail_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "multinli.in.out": {
        "train": os.path.join(
            args.data_dir, "multinli.in.out_to_squad-train-v2.0.json"
        ),
        "eval": os.path.join(args.data_dir, "multinli.in.out_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "multinli.in.out_to_squad-dev-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "sst": {
        "train": os.path.join(args.data_dir, "sst_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "sst_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "sst_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "srl": {
        "train": os.path.join(args.data_dir, "srl_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "srl_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "srl_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "zre": {
        "train": os.path.join(args.data_dir, "zre_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "zre_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "zre_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "woz.en": {
        "train": os.path.join(args.data_dir, "woz.en_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "woz.en_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "woz.en_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "wikisql": {
        "train": os.path.join(args.data_dir, "wikisql_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "wikisql_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "wikisql_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "schema": {
        "train": os.path.join(args.data_dir, "schema_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "schema_to_squad-dev-v2.0.json"),
        "test": os.path.join(args.data_dir, "schema_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "ag": {
        "train": os.path.join(args.data_dir, "ag_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "ag_to_squad-test-v2.0.json"),
        "test": os.path.join(args.data_dir, "ag_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "dbpedia": {
        "train": os.path.join(args.data_dir, "dbpedia_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "dbpedia_to_squad-test-v2.0.json"),
        "test": os.path.join(args.data_dir, "dbpedia_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "yahoo": {
        "train": os.path.join(args.data_dir, "yahoo_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "yahoo_to_squad-test-v2.0.json"),
        "test": os.path.join(args.data_dir, "yahoo_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "amazon": {
        "train": os.path.join(args.data_dir, "amazon_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "amazon_to_squad-test-v2.0.json"),
        "test": os.path.join(args.data_dir, "amazon_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
    "yelp": {
        "train": os.path.join(args.data_dir, "yelp_to_squad-train-v2.0.json"),
        "eval": os.path.join(args.data_dir, "yelp_to_squad-test-v2.0.json"),
        "test": os.path.join(args.data_dir, "yelp_to_squad-test-v2.0.json"),
        "n_train_epochs": args.n_train_epochs,
    },
}
