import csv
import logging
import os
import pathlib
import pickle
from multiprocessing import Pool

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.nn.utils.rnn import pad_sequence

# import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

# from collections import OrderedDict
from tqdm import tqdm

from .pretrain_settings import (
    DECODER_SPECIAL_TOKENS,
    DECODER_TOKENIZER,
    ENCODER_SPECIAL_TOKENS,
    ENCODER_TOKENIZER,
    FILL_VAL,
    LEN_FACTOR,
    args,
)

logger = logging.getLogger(__name__)


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_losses(
    parallel_encoder, parallel_decoder, enc_inputs, dec_inputs, dec_labels, loss_fct
):

    # encode sequence to feature vector
    if not args.only_lm:
        feature = parallel_encoder(enc_inputs)
        if not args.fp32:
            feature = [fea.half() for fea in feature]
        # decode feature to sequence
        ae_logits = parallel_decoder(dec_inputs, feature=feature)

        # calculate loss
        ae_loss = loss_fct(
            [torch.transpose(logit, 1, 2) for logit in ae_logits], dec_labels
        )
        # concat and to cpu memory
        feature = torch.cat([f.cpu() for f in feature])
    else:
        ae_loss = torch.tensor(0.0)
        feature = None

    # language modeling
    if args.with_lm:
        lm_logits = parallel_decoder(dec_inputs)
        lm_loss = loss_fct(
            [torch.transpose(logit, 1, 2) for logit in lm_logits], dec_labels
        )
    else:
        lm_loss = torch.tensor(0.0)

    return torch.mean(ae_loss), torch.mean(lm_loss), feature


def varlen_collate_fn(data):
    enc_pad_id = ENCODER_TOKENIZER.convert_tokens_to_ids(
        ENCODER_SPECIAL_TOKENS["pad_token"]
    )
    dec_pad_id = DECODER_TOKENIZER.convert_tokens_to_ids(
        DECODER_SPECIAL_TOKENS["pad_token"]
    )
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus

    enc_ids_list, dec_ids_list = zip(*data)

    enc_input_ids = pad_sequence(
        enc_ids_list, batch_first=True, padding_value=enc_pad_id
    )
    dec_input_ids = pad_sequence(
        dec_ids_list, batch_first=True, padding_value=dec_pad_id
    )
    dec_label_ids = pad_sequence(dec_ids_list, batch_first=True, padding_value=FILL_VAL)

    return (
        list(enc_input_ids.split(batch_size)),
        list(dec_input_ids.split(batch_size)),
        list(dec_label_ids.split(batch_size)),
    )


def dynamic_collate_fn(data, batch_size):
    def local_collate():
        enc_pad_id = ENCODER_TOKENIZER.convert_tokens_to_ids(
            ENCODER_SPECIAL_TOKENS["pad_token"]
        )
        dec_pad_id = DECODER_TOKENIZER.convert_tokens_to_ids(
            DECODER_SPECIAL_TOKENS["pad_token"]
        )

        enc_ids_list, dec_ids_list = zip(*data)

        enc_input_ids.append(
            pad_sequence(
                enc_ids_list[st:ed], batch_first=True, padding_value=enc_pad_id
            )
        )
        dec_input_ids.append(
            pad_sequence(
                dec_ids_list[st:ed], batch_first=True, padding_value=dec_pad_id
            )[:, :-1]
        )
        dec_label_ids.append(
            pad_sequence(dec_ids_list[st:ed], batch_first=True, padding_value=FILL_VAL)[
                :, 1:
            ]
        )

    enc_input_ids, dec_input_ids, dec_label_ids = [], [], []
    seq_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        seq_len = len(datum[1])
        if max(seq_max_len, seq_len) ** LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            seq_max_len = 0
            st = ed
        seq_max_len = max(seq_max_len, seq_len)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return enc_input_ids, dec_input_ids, dec_label_ids


class PretrainDataset(Dataset):
    def __init__(
        self, text_data=None, data_path=None, data_type="train", source="wiki40b"
    ):
        self.data_type = data_type
        self.cls_token_id = ENCODER_TOKENIZER.cls_token_id
        self.gen_token_id = DECODER_TOKENIZER.convert_tokens_to_ids(
            DECODER_SPECIAL_TOKENS["gen_token"]
        )
        self.eos_token_id = DECODER_TOKENIZER.convert_tokens_to_ids(
            DECODER_SPECIAL_TOKENS["eos_token"]
        )

        if data_path is None:
            data_path = os.path.join(
                args.pretrain_data_dir, source, f"tokenized_{data_type}_data.pkl"
            )

        if os.path.exists(data_path):
            logger.info(f"tokenized data already exists in {data_path}, read it !!")
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            assert text_data is not None
            data = self.tokenize_data(text_data)
            logger.info(f"writing tokenized data in {data_path} ...")
            with open(data_path, "wb") as f:
                pickle.dump(data, f)

        self.data = self.filter_by_seq_len(data)

    def tokenize_sample(self, text):
        enc_input_ids = [self.cls_token_id] + ENCODER_TOKENIZER.encode(text)
        dec_input_ids = (
            [self.gen_token_id] + DECODER_TOKENIZER.encode(text) + [self.eos_token_id]
        )
        return enc_input_ids, dec_input_ids

    def tokenize_data(self, text_data):
        logger.info("tokenizing text data ...")
        with Pool(args.n_workers) as pool:
            data = pool.map(self.tokenize_sample, text_data)
        return data

    def filter_by_seq_len(self, data):
        def len_filter(d):
            return (
                args.min_seq_len <= len(d[0]) <= args.encoder_max_len
                and args.min_seq_len <= len(d[1]) <= args.decoder_max_len
            )

        data = list(filter(len_filter, data))
        return data

    def __getitem__(self, index):
        encoder_ids, decoder_ids = self.data[index]
        return torch.LongTensor(encoder_ids), torch.LongTensor(decoder_ids)

    def __len__(self):
        return len(self.data)


def get_raw_text_data():
    raw_text_data_path = os.path.join(
        args.pretrain_data_dir, "wiki40b", "all_test_text.csv"
    )
    raw_text_data = []
    if os.path.exists(raw_text_data_path):
        with open(raw_text_data_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for text in reader:
                raw_text_data.append(text[0].strip())
    else:
        assert False
        raw_text_data = download_wiki40b_data()
        with open(raw_text_data_path, "w") as f:
            writer = csv.writer(f, delimiter=",")
            for text in raw_text_data:
                writer.writerow([text])
    return raw_text_data


def download_wiki40b_data(data_dir="./", wiki40b_type="test"):
    """Load data from source and save preprocessed data to file"""
    print("Downloading ...")
    tf_data = tfds.load("wiki40b/en", split=wiki40b_type, data_dir=data_dir)
    all_text_data = []
    with tqdm(tf_data.as_numpy_iterator(), ncols=100) as pbar:
        for wiki in pbar:
            for text in (
                wiki["text"]
                .decode()
                .replace("_NEWLINE_", "\n_START_PARAGRAPH_\n")
                .split("\n_START_")
            ):
                key = "PARAGRAPH_\n"
                if text[: len(key)] == key:
                    all_text_data.append(text[len(key) :])
    return all_text_data


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if args.debug or self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][1])
            if max(max_len, ln) ** LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed + 1
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):

        def collate_fn(x, bs=batch_size):
            return dynamic_collate_fn(x, bs)

        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:

        def collate_fn(x):
            return varlen_collate_fn(x)

        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    dataloader = DataLoader(
        dataset,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        batch_size=batch_size,
        batch_sampler=batch_sampler,
    )
    return dataloader


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        return outputs[0]


class TrainStep:
    def __init__(self, encoder, decoder, optimizer, scheduler):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, loss, scheduler_steps):
        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), args.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), args.max_grad_norm
            )

        self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()
