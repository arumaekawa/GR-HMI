import csv
import io
import json
import logging
import os
import pathlib
import random
import re
import sys
import uuid
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import quadprog
import torch
import torch.nn.functional as F
from settings import (
    DATA_ATTRS,
    DECODER_CONFIG,
    DECODER_SPECIAL_TOKEN_IDS,
    DECODER_SPECIAL_TOKENS,
    DECODER_TOKENIZER,
    ENCODER_SPECIAL_TOKEN_IDS,
    ENCODER_TOKENIZER,
    FILL_VAL,
    LEN_FACTOR,
    LLL_METHODS,
    MEMORY_FACTOR,
    TASK_DICT,
    args,
)
from torch.utils.data import DataLoader, Dataset, Sampler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_gen_token(task):
    if args.add_task_tokens:
        return "__" + task + "__"
    else:
        return "__gen__"


def get_model_dir(tasks):
    if args.seq_train_type != "multitask":
        return os.path.join(args.model_dir_root, tasks[0])
    else:
        return args.model_dir_root


# def get_losses(parallel_encoder, parallel_decoder,
#                cls_cqa, cqa, Y, gen_X, gen_Y, loss_fct):
#     # task loss (as question answering task)
#     qa_loss = get_qa_loss(parallel_decoder, cqa, Y, loss_fct)
#     # sample generation loss (as autoencoder or language modeling)
#     if args.seq_train_type == "hmi-lamol":
#         gen_loss, feature = get_ae_loss(
#             parallel_encoder, parallel_decoder, cls_cqa, gen_X, gen_Y,
#             loss_fct)
#     elif args.seq_train_type == "lamol":
#         gen_loss = get_lm_loss(
#             parallel_decoder, gen_X, gen_Y, loss_fct)
#     else:
#         gen_loss = torch.tensor(0.)
#     # total loss
#     losses = (qa_loss, args.gen_lambda * gen_loss)
#     return losses, feature


def get_losses(
    parallel_decoder,
    cqa,
    qa_Y,
    gen_Y,
    loss_fct,
    parallel_encoder=None,
    cls_cq=None,
    hc_module=None,
):

    """ae losses for `hmi-lamol`"""
    if args.seq_train_type == "hmi-lamol":
        assert parallel_encoder is not None and cls_cq is not None

        # encode examples to feature vectors
        feature = parallel_encoder(cls_cq)
        if not args.fp32:
            feature = [fea.half() for fea in feature]

        if "pq" in args.hc_module_type:
            assert not args.train_encoder
            feature = [
                hc_module.reform_feature(fea.cpu()).to(device_id)
                for fea, device_id in zip(feature, args.device_ids)
            ]
    else:
        feature = [None] * len(args.device_ids)

    # forward (question answering)
    logits = parallel_decoder(cqa, feature=feature)
    # compute question answering loss
    qa_loss = loss_fct([torch.transpose(logit, 1, 2) for logit in logits], qa_Y)
    if args.seq_train_type in LLL_METHODS:
        gen_loss = loss_fct([torch.transpose(logit, 1, 2) for logit in logits], gen_Y)
    else:
        gen_loss = torch.tensor(0.0)

    return torch.mean(qa_loss), torch.mean(gen_loss)


def pad_to_max_len(seq, pad_len, val):
    return seq + [val] * pad_len


def pad_all_to_max_len(batch, val):
    max_len = max(len(seq) for seq in batch)
    return [pad_to_max_len(seq, max_len - len(seq), val) for seq in batch]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # # scatter sorted tensors to original indexing
        # indices_to_remove = sorted_indices_to_remove.scatter(
        #     1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(
        pad_all_to_max_len(
            [datum["dec_examples"]["cq"] for datum in data],
            DECODER_SPECIAL_TOKEN_IDS["pad_token"],
        )
    ).split(batch_size)
    len_cqs = torch.tensor([datum["dec_examples"]["len_cq"] for datum in data]).split(
        batch_size
    )
    cqas = torch.tensor(
        pad_all_to_max_len(
            [datum["dec_examples"]["cqa"] for datum in data],
            DECODER_SPECIAL_TOKEN_IDS["pad_token"],
        )
    ).split(batch_size)
    len_cqas = torch.tensor([datum["dec_examples"]["len_cqa"] for datum in data]).split(
        batch_size
    )
    qa_Ys = torch.tensor(
        pad_all_to_max_len([datum["dec_examples"]["qa_Y"] for datum in data], FILL_VAL)
    ).split(batch_size)
    gen_Ys = torch.tensor(
        pad_all_to_max_len([datum["dec_examples"]["gen_Y"] for datum in data], FILL_VAL)
    ).split(batch_size)
    is_replays = torch.tensor([datum["is_replay"] for datum in data]).split(batch_size)

    if data[0]["enc_examples"] is not None:
        cls_cq = torch.tensor(
            pad_all_to_max_len(
                [datum["enc_examples"]["cls_cq"] for datum in data],
                ENCODER_SPECIAL_TOKEN_IDS["pad_token"],
            )
        ).split(batch_size)
        return (
            list(cls_cq),
            list(cqs),
            list(len_cqs),
            list(cqas),
            list(len_cqas),
            list(qa_Ys),
            list(gen_Ys),
            list(is_replays),
        )

    return (
        None,
        list(cqs),
        list(len_cqs),
        list(cqas),
        list(len_cqas),
        list(qa_Ys),
        list(gen_Ys),
        list(is_replays),
    )


def dynamic_collate_fn(data, batch_size):
    def local_collate():
        null_counter = 0
        _cqs, _len_cqs, _cqas, _len_cqas, _qa_Ys, _gen_Ys, _is_replay = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        Y_max_len = max(len(data[j]["dec_examples"]["qa_Y"]) for j in range(st, ed))
        cq_max_len = max(len(data[j]["dec_examples"]["cq"]) for j in range(st, ed))
        if use_enc_examples:
            _cls_cqs = []
            enc_max_len = max(
                [len(data[j]["enc_examples"]["cls_cq"]) for j in range(st, ed)]
            )
        for j in range(st, ed):
            if (
                None in data[j]["dec_examples"].values()
                or [] in data[j]["dec_examples"].values()
            ):
                null_counter += 1
                logger.warning(
                    "null example in collate_fn, count: {}".format(null_counter)
                )
                continue

            if use_enc_examples:
                # enc_examples pad len
                enc_pad_len = enc_max_len - len(data[j]["enc_examples"]["cls_cq"])
                # add pad ids to enc_example
                _cls_cqs.append(
                    pad_to_max_len(
                        data[j]["enc_examples"]["cls_cq"],
                        enc_pad_len,
                        ENCODER_SPECIAL_TOKEN_IDS["pad_token"],
                    )
                )

            # dec_examples pad len
            dec_pad_len = cqa_max_len - data[j]["dec_examples"]["len_cqa"]

            # add pad ids to each dec_examples
            _cqs.append(
                pad_to_max_len(
                    data[j]["dec_examples"]["cq"],
                    cq_max_len - data[j]["dec_examples"]["len_cq"],
                    DECODER_SPECIAL_TOKEN_IDS["pad_token"],
                )
            )
            _len_cqs.append(data[j]["dec_examples"]["len_cq"])
            _cqas.append(
                pad_to_max_len(
                    data[j]["dec_examples"]["cqa"],
                    dec_pad_len,
                    DECODER_SPECIAL_TOKEN_IDS["pad_token"],
                )
            )
            _len_cqas.append(data[j]["dec_examples"]["len_cqa"])
            _qa_Ys.append(
                pad_to_max_len(
                    data[j]["dec_examples"]["qa_Y"],
                    Y_max_len - len(data[j]["dec_examples"]["qa_Y"]),
                    FILL_VAL,
                )
            )
            _gen_Ys.append(
                pad_to_max_len(data[j]["dec_examples"]["gen_Y"], dec_pad_len, FILL_VAL)
            )
            _is_replay.append(data[j]["is_replay"])

        if use_enc_examples:
            cls_cqs.append(torch.tensor(_cls_cqs))
        cqs.append(torch.tensor(_cqs))
        len_cqs.append(torch.tensor(_len_cqs))
        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        qa_Ys.append(torch.tensor(_qa_Ys))
        gen_Ys.append(torch.tensor(_gen_Ys))
        is_replays.append(torch.tensor(_is_replay))

    use_enc_examples = data[0]["enc_examples"] is not None
    cls_cqs = [] if use_enc_examples else None
    cqs, len_cqs, cqas, len_cqas, qa_Ys, gen_Ys, is_replays = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cqa_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        ln = len(datum["dec_examples"]["cqa"])  # use cqas to calibrate
        if max(cqa_max_len, ln) ** LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return (cls_cqs, cqs, len_cqs, cqas, len_cqas, qa_Ys, gen_Ys, is_replays)


class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, extra_data=[]):
        self.data_type = data_type
        self.use_encoder = args.seq_train_type == "hmi-lamol"
        self.gen_token = gen_token
        if args.use_sep:
            self.dec_sep_token = DECODER_SPECIAL_TOKEN_IDS["sep_token"]
        self.dec_ans_token = DECODER_SPECIAL_TOKEN_IDS["ans_token"]
        self.dec_eos_token = DECODER_SPECIAL_TOKEN_IDS["eos_token"]
        self.dec_pad_token = DECODER_SPECIAL_TOKEN_IDS["pad_token"]
        if self.use_encoder:
            if args.use_sep:
                self.enc_sep_token = ENCODER_SPECIAL_TOKEN_IDS["sep_token"]
            self.cls_token = ENCODER_SPECIAL_TOKEN_IDS["cls_token"]
            self.enc_ans_token = ENCODER_SPECIAL_TOKEN_IDS["ans_token"]
            self.enc_pad_token = ENCODER_SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d

        self.data = []
        self.max_a_len = 0
        if (
            len(data_paths) == 1
            and data_paths[0] is not None
            and ("wiki" in data_paths[0] or "woz" in data_paths[0])
        ):
            # data = self._sort_by_index(data)
            # args.n_workers = 1
            if "wiki" in data_paths[0]:
                answers_file = "wikisql_answers.json"
            elif "woz" in data_paths[0]:
                answers_file = "woz.en_answers.json"
            with open(os.path.join(args.data_dir, answers_file), "r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)

        if len(extra_data) > 0:
            num_all_extra_data = len(extra_data)
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x: x, extra_data))
            if args.gen_lm_sample_percentage > 0.0 and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            else:
                logger.info(
                    "Number of good extra data {} / {}".format(
                        len(extra_data), num_all_extra_data
                    )
                )
            self.data += extra_data

    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = " ".join([str(datum) for datum in data[1:]])
        try:
            if args.use_sep:
                dec_context, dec_qa = re.split(
                    str(DECODER_SPECIAL_TOKEN_IDS["sep_token"]), data
                )
            else:
                dec_context = ""
                dec_qa = data
            dec_question, dec_answer = re.split(
                str(DECODER_SPECIAL_TOKEN_IDS["ans_token"]), dec_qa
            )
            dec_context = [int(c) for c in dec_context.strip().split()]
            dec_question = [int(q) for q in dec_question.strip().split()]
            dec_answer = [
                int(a)
                for a in re.sub(
                    str(DECODER_SPECIAL_TOKEN_IDS["eos_token"]), "", dec_answer
                )
                .strip()
                .split()
            ]
            if self.use_encoder:

                def dec_ids_to_enc_ids(dec_ids):
                    return ENCODER_TOKENIZER.encode(DECODER_TOKENIZER.decode(dec_ids))

                enc_context, enc_question, enc_answer = map(
                    dec_ids_to_enc_ids, (dec_context, dec_question, dec_answer)
                )
                enc_examples = self.parse_enc_example(
                    self.cls_token, enc_context, enc_question, enc_answer
                )
                if enc_examples is None:
                    raise ValueError
            else:
                enc_examples = None
            dec_examples = self.parse_dec_example(
                gen_token, dec_context, dec_question, dec_answer
            )
            uid = uuid.uuid1().hex
        except ValueError:
            return
        return {
            "enc_examples": enc_examples,
            "dec_examples": dec_examples,
            "is_replay": True,
            "id": uid,
        }

    def enc_concat_example(self, cls_token, c, sep_token, q, ans_token, a):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.encoder_max_len:
            logger.warning(
                "an example with len {} is too long!".format(len(example) + 1)
            )
            raise ValueError
        example = cls_token + c[: args.encoder_max_len - len(example) - 1] + example
        return example

    def dec_concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.decoder_max_len:
            logger.warning(
                "an example with len {} is too long!".format(len(example) + 1)
            )
            raise ValueError
        example = (
            gen_token
            + c[: args.decoder_max_len - len(example) - 1]
            + example
            + eos_token
        )
        return example

    def parse_enc_example(self, cls_token, context, question, answer):
        if args.use_sep:
            cls_cq = self.enc_concat_example(
                [cls_token], context, [self.enc_sep_token], question, [], []
            )
        else:
            cls_cq = self.enc_concat_example([cls_token], context, [], question, [], [])
        return {"cls_cq": cls_cq}

    def parse_dec_example(self, gen_token, context, question, answer):
        if args.use_sep:
            cq_example = self.dec_concat_example(
                [gen_token],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                [],
                [],
            )
            cqa_example = self.dec_concat_example(
                [gen_token],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                answer,
                [],
            )
        else:
            cq_example = self.dec_concat_example(
                [gen_token], context, [], question, [self.dec_ans_token], [], []
            )
            cqa_example = self.dec_concat_example(
                [gen_token], context, [], question, [self.dec_ans_token], answer, []
            )
        qa_Y_example = self.dec_concat_example(
            [], [], [], [], [], answer, [self.dec_eos_token]
        )  # shifted
        qa_Y_example = [FILL_VAL] * (
            len(cqa_example) - len(qa_Y_example)
        ) + qa_Y_example
        if args.use_sep:
            gen_Y_example = self.dec_concat_example(
                [],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                answer,
                [self.dec_eos_token],
            )
        else:
            gen_Y_example = self.dec_concat_example(
                [],
                context,
                [],
                question,
                [self.dec_ans_token],
                answer,
                [self.dec_eos_token],
            )
        return {
            "cq": cq_example,
            "len_cq": len(cq_example),
            "cqa": cqa_example,
            "len_cqa": len(cqa_example),
            "qa_Y": qa_Y_example,
            "gen_Y": gen_Y_example,
        }

    def parallel_tokenization(self, d):
        examples = []
        dec_context = DECODER_TOKENIZER.encode(d["context"])
        if self.use_encoder:
            enc_context = ENCODER_TOKENIZER.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            if self.use_encoder:
                enc_question = ENCODER_TOKENIZER.encode(qa["question"])
            dec_question = DECODER_TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            if self.use_encoder:
                enc_answer = []
            dec_answer = []
            for i, raw_answer in enumerate(raw_answers):
                if self.use_encoder:
                    enc_answer.extend(ENCODER_TOKENIZER.encode(raw_answer["text"]))
                dec_answer.extend(DECODER_TOKENIZER.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    if self.use_encoder:
                        enc_answer.append(self.enc_pad_token)
                    dec_answer.append(self.dec_pad_token)
            max_a_len = max(max_a_len, len(dec_answer))

            if self.use_encoder:
                enc_examples = self.parse_enc_example(
                    self.cls_token, enc_context, enc_question, enc_answer
                )
            else:
                enc_examples = None

            dec_examples = self.parse_dec_example(
                self.gen_token, dec_context, dec_question, dec_answer
            )

            examples.append(
                {
                    "enc_examples": enc_examples,
                    "dec_examples": dec_examples,
                    "is_replay": False,
                    "id": qa.get("id", 0),
                }
            )

        return examples, max_a_len

    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: x["dec_examples"]["len_cq"])
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x["id"])

    def get_indices(self):
        return [d["id"] for d in self.data]

    # def _sort_by_index(self,data):
    #    datum = []
    #    for d in data:
    #        for qa in d["qas"]:
    #            datum.append({"context":d["context"], "qas":[qa]})
    #    datum.sort(key=lambda x:x["qas"][0]["id"])
    #    return datum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class EarlyStopping:
    def __init__(self, logger, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.logger.info(
                "Validation loss decreased ({:.6f} "
                "--> {:.6f}).  Saving model ...".format(self.val_loss_min, val_loss)
            )
        model.save_pretrained(model_dir)
        DECODER_TOKENIZER.save_pretrained(model_dir)
        self.val_loss_min = val_loss


class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

        if "gem" in args.seq_train_type and self.model.task_id > 0:
            store_grad(
                self.model.parameters,
                self.model.grads,
                self.model.grad_dims,
                self.model.task_id,
            )
            indx = torch.cuda.LongTensor([i for i in range(self.model.task_id)])
            dotp = torch.mm(
                self.model.grads[:, self.model.task_id].unsqueeze(0),
                self.model.grads.index_select(1, indx),
            )
            if (dotp < 0).sum() != 0:
                project2cone2(
                    self.model.grads[:, self.model.task_id].unsqueeze(1),
                    self.model.grads.index_select(1, indx),
                    args.qp_margin,
                )
                # copy gradients back
                overwrite_grad(
                    self.model.parameters,
                    self.model.grads[:, self.model.task_id],
                    self.model.grad_dims,
                )

        if args.seq_train_type in args.REG_TYPE_KEYS:
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()


class GEMStep:
    def __init__(self, model, parallel_model, train_loss_fct, optimizer):
        self.model = model
        self.parallel_model = parallel_model
        self.train_loss_fct = train_loss_fct
        self.optimizer = optimizer

    def __call__(self, current_task_id):
        raise NotImplementedError
        # for past_task_id, md in enumerate(args.memory_data):
        #     # Not saving current task's grads.
        #     if past_task_id >= current_task_id:
        #         return
        #     qadata = QADataset(None, "test", "gen", md)
        #     dataloader = create_dataloader(qadata, "test")
        #     grads_tmp = torch.zeros(
        #         sum(self.model.grad_dims),
        #     ).cuda()
        #     if not args.fp32:
        #         grads_tmp = grads_tmp.half()
        #     for _, _, _, cqa, _, Y, _, gen_Y, _ in dataloader:
        #         # CHECK
        #         n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
        #         self.optimizer.zero_grad()
        #         # for i in range(len(cqa)):
        #         #     cqa[i] = (cqa[i].to(args.device_ids[i]),)
        #         #     Y[i] = Y[i].to(args.device_ids[i])
        #         #     gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
        #         #     gen_Y[i] = gen_Y[i].to(args.device_ids[i])

        #         # losses = get_losses(self.parallel_model, cqa,
        #         #                     Y, gen_X, gen_Y, self.train_loss_fct)
        #         # loss = sum(losses)
        #         for i in range(len(cqa)):
        #             cqa[i] = (cqa[i].to(args.device_ids[i]),)
        #             Y[i] = Y[i].to(args.device_ids[i])
        #             cqa[i] = (cqa[i].to(args.device_ids[i]),)
        #             gen_Y[i] = gen_Y[i].to(args.device_ids[i])

        #         # task loss (as question answering task)
        #         qa_loss = get_qa_loss(
        #             self.parallel_decoder, cqa, Y, self.train_loss_fct
        #         )

        #         # sample generation loss (as autoencoder or language modeling)
        #         if args.seq_train_type == "lamol":
        #             gen_loss = get_lm_loss(
        #                 self.parallel_decoder, cqa, gen_Y, self.train_loss_fct
        #             )
        #         else:
        #             gen_loss = torch.tensor(0.0)

        #         # total loss
        #         loss = qa_loss + args.gen_lambda * gen_loss
        #         if not args.fp32:
        #             self.optimizer.backward(loss, update_master_grads=False)
        #         else:
        #             loss.backward()

        #         if not args.fp32:
        #             # copy fp16 grads to fp32 grads
        #             self.optimizer.update_master_grads()
        #             self.optimizer.clip_master_grads(args.max_grad_norm)
        #         else:
        #             torch.nn.utils.clip_grad_norm_(
        #                 self.model.parameters(), args.max_grad_norm
        #             )
        #         i = 0
        #         for param in self.model.parameters():
        #             if param.grad is not None:
        #                 beg = 0 if i == 0 else sum(self.model.grad_dims[:i])
        #                 end = sum(self.model.grad_dims[: i + 1])
        #                 grads_tmp[beg:end] += param.grad.data.view(-1) * n_inputs
        #             i += 1

        #     grads_tmp /= len(qadata)
        #     self.model.grads[:, past_task_id].copy_(grads_tmp)
        #     self.optimizer.zero_grad()


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
            ln = self.dataset[idx]["dec_examples"]["len_cqa"]
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


def add_all_feature_to_hc_memory(train_data, parallel_encoder, hc_module):
    dataloader = DataLoader(
        train_data,
        num_workers=args.n_workers,
        collate_fn=varlen_collate_fn,
        shuffle=False,
        batch_size=48,
    )
    with torch.no_grad():
        for (cls_cqs, _, _, cqas, len_cqas, _, _, is_replays) in dataloader:
            cls_cqs = [
                (d.to(device_id),) for d, device_id in zip(cls_cqs, args.device_ids)
            ]
            # encode to features
            feature = parallel_encoder(cls_cqs)
            feature = torch.cat([fea.cpu() for fea in feature])
            if not args.fp32:
                feature = feature.half()
            # add current task's features to hippocampus module
            hc_module.memorize(feature[torch.cat(is_replays).logical_not()].detach())
            if "real" in args.hc_module_type:
                all_cqas, all_len_cqas = [], []
                for cqa, len_cqa, is_replay in zip(cqas, len_cqas, is_replays):
                    all_cqas += list(cqa[is_replay.logical_not()])
                    all_len_cqas += list(len_cqa[is_replay.logical_not()])
                hc_module.memorize_real_sample(
                    [
                        cqa[:len_cqa].cpu()
                        for cqa, len_cqa in zip(all_cqas, all_len_cqas)
                    ]
                )


def get_loss_of_all_train_data(
    train_data,
    parallel_encoder,
    parallel_decoder,
    loss_fct,
    hc_module,
    loss_type="both",
):

    all_losses = []
    data_loader = DataLoader(
        train_data,
        num_workers=args.n_workers,
        collate_fn=varlen_collate_fn,
        shuffle=False,
        batch_size=16,
    )
    with torch.no_grad():
        for (cls_cq, _, _, cqa, _, qa_Y, gen_Y, is_replays) in data_loader:

            for i in range(len(cqa)):
                if cls_cq is not None:
                    cls_cq[i] = (cls_cq[i].to(args.device_ids[i]),)
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                qa_Y[i] = qa_Y[i].to(args.device_ids[i])
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])

            # task loss (as question answering task)
            if args.seq_train_type == "hmi-lamol":
                assert parallel_encoder is not None and cls_cq is not None

                # encode examples to feature vectors
                feature = parallel_encoder(cls_cq)
                if not args.fp32:
                    feature = [fea.half() for fea in feature]

                if "pq" in args.hc_module_type:
                    assert not args.train_encoder
                    feature = [
                        hc_module.reform_feature(fea.cpu()).to(device_id)
                        for fea, device_id in zip(feature, args.device_ids)
                    ]
            else:
                feature = [None] * len(args.device_ids)

            # forward (question answering)
            logits = parallel_decoder(cqa, feature=feature)
            # compute question answering loss
            assert all(
                [logit.ndim == 3 for logit in logits]
            ), f"logits shape: {[logit.shape for logit in logits]}"

            qa_losses = torch.cat(
                [
                    F.cross_entropy(
                        torch.transpose(logit, 1, 2),
                        qa_y,
                        ignore_index=FILL_VAL,
                        weight=loss_fct.module.weight,
                        reduction="none",
                    ).mean(1)
                    for logit, qa_y in zip(logits, qa_Y)
                ]
            )
            gen_losses = torch.cat(
                [
                    F.cross_entropy(
                        torch.transpose(logit, 1, 2),
                        gen_y,
                        ignore_index=FILL_VAL,
                        weight=loss_fct.module.weight,
                        reduction="none",
                    ).mean(1)
                    for logit, gen_y in zip(logits, gen_Y)
                ]
            )
            if loss_type == "both":
                losses = qa_losses + args.gen_lambda * gen_losses
            elif loss_type == "qa":
                losses = qa_losses
            elif loss_type == "gen":
                losses = gen_losses
            else:
                assert False

            all_losses += losses[torch.cat(is_replays).logical_not()].tolist()

    return all_losses


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids, **kwargs):
        outputs = self.model(input_ids, **kwargs)
        return outputs[0]


def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(DECODER_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0


def sample_sequence(
    model,
    need_process,
    qa_results,
    all_pasts,
    max_tot_lens,
    features=None,
    top_k=0,
    top_p=0.0,
):
    while len(need_process) > 0:
        first_id = next(iter(need_process))
        shortest_len = len(qa_results[first_id])
        decode_batch_size = int(
            args.memory_sizes[0]
            * MEMORY_FACTOR[args.seq_train_type]
            // (shortest_len + 1) ** LEN_FACTOR
        )
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, feature, past = (
                [],
                [],
                [],
                [[] for _ in range(DECODER_CONFIG.n_layer)],
            )
            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    if args.decoder_model_name == "gpt2":
                        input_ids.append(qa_results[cur_id][-1:])
                        for layer_id in range(DECODER_CONFIG.n_layer):
                            past[layer_id].append(all_pasts[layer_id][cur_id])
                    else:
                        input_ids.append(qa_results[cur_id])
                    if features is not None:
                        feature.append(features[cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            feature = torch.stack(feature) if feature else None
            if args.decoder_model_name == "gpt2":
                for layer_id in range(DECODER_CONFIG.n_layer):
                    past[layer_id] = torch.stack(past[layer_id], dim=1)
                all_outputs = model(
                    input_ids=input_ids.cuda(), feature=feature, past=past
                )
            else:
                all_outputs = model(input_ids=input_ids.cuda(), feature=feature)

            outputs = all_outputs[0]
            if args.decoder_model_name == "gpt2":
                pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits, top_k=top_k, top_p=top_p).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == DECODER_SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [
                        max_tot_lens[cur_id],
                        args.decoder_max_len,
                    ]:
                        remove_ids.append(cur_id)
                    elif args.decoder_model_name == "gpt2":
                        for layer_id in range(DECODER_CONFIG.n_layer):
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(
                                torch.float if args.fp32 else torch.half
                            )
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)


def write_extra_data(dump_path, qa_results):
    logger.info(f"writing extra data in {dump_path} ...")
    with open(dump_path, "w", newline="", encoding="utf-8") as f:
        lm_writer = csv.writer(f, delimiter=",")
        lm_writer.writerow(["gen"])
        for line in qa_results:
            lm_writer.writerow([line])


def parse_single_real_data(data, task):
    c = data["paragraphs"][0]["context"]
    q = data["paragraphs"][0]["qas"][0]["question"]
    a = data["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    if args.use_sep:
        data = "{}{}{}{}{}{}{}".format(
            DECODER_SPECIAL_TOKENS[task],
            c,
            DECODER_SPECIAL_TOKENS["sep_token"],
            q,
            DECODER_SPECIAL_TOKENS["ans_token"],
            a,
            DECODER_SPECIAL_TOKENS["eos_token"],
        )
    else:
        data = "{}{} {}{}{}{}".format(
            DECODER_SPECIAL_TOKENS[task],
            c,
            q,
            DECODER_SPECIAL_TOKENS["ans_token"],
            a,
            DECODER_SPECIAL_TOKENS["eos_token"],
        )
    return data


def get_real_data(task, train_extra_data, accum=True, encode=True):
    task_idx = args.tasks.index(task)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    if accum:
        prev_tasks = args.tasks[:task_idx]
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage)) // len(
            prev_tasks
        )
    else:
        prev_tasks = [args.tasks[task_idx - 1]]
        gen_size = int(gen_size * args.gen_lm_sample_percentage)

    datum = []
    for prev_task in prev_tasks:
        with open(TASK_DICT[prev_task]["train"], "r") as f:
            data = data_expand(json.load(f)["data"])
        indices = np.random.choice(range(len(data)), gen_size)
        for i in indices:
            d = parse_single_real_data(data[i], prev_task)
            datum.append(d)
            if encode:
                train_extra_data.append(DECODER_TOKENIZER.encode(d))

    model_dir = get_model_dir([prev_task])
    dump_path = os.path.join(model_dir, "real.csv")
    write_extra_data(dump_path, datum)
    return dump_path


def read_extra_data(gen_path, train_extra_data):
    with open(gen_path, "r") as lm_file:
        reader = csv.reader(lm_file, delimiter=",")
        next(reader)
        for row in reader:
            row = DECODER_TOKENIZER.encode(row[0].strip())
            train_extra_data.append(row)


def get_hc_module_memory(hc_module, cur_task):
    # cur_task_id
    cur_task_id = args.tasks.index(cur_task)
    # previous tasks
    prev_tasks = args.tasks[:cur_task_id]

    # number of replay samples for each task
    gen_size = DATA_ATTRS[cur_task]["train"]["data_size"]
    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage)) // len(prev_tasks)

    # retrieve from hippocampus module
    if "real" in args.hc_module_type:
        real_samples = []
        for prev_task in prev_tasks:
            real_samples += hc_module.retrieve(prev_task, gen_size)
        return real_samples
    else:
        features = []
        for prev_task in prev_tasks:
            features_of_one_task = hc_module.retrieve(prev_task, gen_size)
            features.append(features_of_one_task)
        # concat features as tensor
        features = torch.cat(features)

        return features


def create_extra_data(task, prev_task, model, train_extra_data, hc_memory=None):
    if args.real_sample:
        logger.info("using real data as extra data")
        return get_real_data(task, train_extra_data)
    task_cnt = args.tasks.index(task)
    model_dir = get_model_dir([prev_task])
    gen_path = os.path.join(model_dir, "lm.csv")
    if os.path.exists(gen_path):
        logger.info(f"extra data exists in {gen_path}, read it!")
        return read_extra_data(gen_path, train_extra_data)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]

    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    gen_size -= gen_size % task_cnt
    if hc_memory is not None:
        assert gen_size == len(hc_memory)

    if "real" in args.hc_module_type:
        qa_results = hc_memory
    else:
        features = hc_memory

        if args.debug:
            gen_size = task_cnt

        logger.info("generate replay samples ...")

        model.eval()

        need_process = OrderedDict()
        qa_results = []
        for task_name in args.tasks[:task_cnt]:
            # bos_token_id
            qa_results.extend(
                [
                    torch.tensor([DECODER_SPECIAL_TOKEN_IDS[task_name]])
                    for _ in range(gen_size // task_cnt)
                ]
            )
        if features is not None:
            features = [fea.cuda() for fea in features]
        all_pasts = [
            [
                torch.empty(
                    2,
                    DECODER_CONFIG.n_head,
                    0,
                    DECODER_CONFIG.n_embd // DECODER_CONFIG.n_head,
                    dtype=torch.float if args.fp32 else torch.half,
                ).cuda()
                for _ in range(gen_size)
            ]
            for __ in range(DECODER_CONFIG.n_layer)
        ]
        max_tot_len = (
            args.decoder_max_len
            if args.seq_train_type != "hmi-lamol"
            else min(args.encoder_max_len, args.decoder_max_len)
        )
        max_tot_lens = [max_tot_len for _ in range(gen_size)]

        for i in range(gen_size):
            need_process.update([[i, None]])
            if len(need_process) > int(args.memory_sizes[0] * 0.12):
                sample_sequence(
                    model,
                    need_process,
                    qa_results,
                    all_pasts,
                    max_tot_lens,
                    features=features,
                    top_k=args.top_k_lm,
                )
        sample_sequence(
            model,
            need_process,
            qa_results,
            all_pasts,
            max_tot_lens,
            features=features,
            top_k=args.top_k_lm,
        )

        model.train()

    qa_results = [res.tolist() for res in qa_results]
    train_extra_data.extend(qa_results)
    qa_results = [DECODER_TOKENIZER.decode(res) for res in qa_results]

    write_extra_data(gen_path, qa_results)


def logits_to_tokens(next_logits, top_k=0, top_p=0.0):
    if top_k == 1:
        # greedy decoding
        next_tokens = next_logits.argmax(-1, keepdim=True)
        return next_tokens
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens


def lll_unbound_setting(split_size=10, data_type="train", test_target="self"):
    data_dir = os.path.join(
        args.data_dir,
        "{}_{}".format("_".join(args.tasks), args.gen_lm_sample_percentage),
    )
    if data_type == "test":
        args.splitted_tasks = [f"task_{i}" for i in range(split_size)]
        args.n_train_epochs = {
            task: args.n_train_epochs for task in args.splitted_tasks
        }
        if test_target in ["self", "all"]:
            for no in range(split_size):
                task = f"task_{no}"
                test_data_path = os.path.join(data_dir, f"{task}-test.json")
                TASK_DICT[task] = {}
                TASK_DICT[task]["test"] = test_data_path
            if test_target == "all":
                args.tasks += args.splitted_tasks
            else:
                args.tasks = args.splitted_tasks
    elif data_type == "train":
        create_lll_unbound_data(split_size)
        args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return TASK_DICT


def create_lll_unbound_data(split_size=10):
    data_dir = os.path.join(
        args.data_dir,
        "{}_{}".format("_".join(args.tasks), args.gen_lm_sample_percentage),
    )
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    datum = []
    test_datum = []
    data_sizes = []
    chunk_sizes = []
    for task in args.tasks:
        train_data_path = TASK_DICT[task]["train"]
        with open(train_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            data_sizes.append(len(data))
            datum += data
        test_data_path = TASK_DICT[task]["test"]
        with open(test_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            test_datum.append(data)
    chunk_size = int(np.ceil(len(datum) / split_size))

    tasks = []
    for no, i in enumerate(range(0, len(datum), chunk_size)):
        task = f"task_{no}"
        tasks.append(task)
        chunk = datum[i : i + chunk_size] if i < len(datum) - chunk_size else datum[i:]
        chunk_sizes.append(len(chunk))
        DATA_ATTRS[task] = {"train": {"data_size": None}}
        DATA_ATTRS[task]["train"]["data_size"] = len(chunk)
        train_data_path = os.path.join(data_dir, f"{task}-train.json")
        with open(train_data_path, "w") as f:
            json.dump({"data": chunk}, f)
        TASK_DICT[task] = {}
        TASK_DICT[task]["train"] = train_data_path
    args.tasks = tasks

    sis = get_split_indices(data_sizes, chunk_sizes)
    test_split = []
    for dic in sis.values():
        merged_data = []
        for k, v in dic.items():
            from_index = int(len(test_datum[k]) * v[0])
            to_index = int(len(test_datum[k]) * v[1])
            merged_data += test_datum[k][from_index:to_index]
        test_split.append(merged_data)

    for no, chunk in enumerate(test_split):
        task = f"task_{no}"
        test_data_path = os.path.join(data_dir, f"{task}-test.json")
        with open(test_data_path, "w") as f:
            json.dump({"data": chunk}, f)
        TASK_DICT[task]["test"] = test_data_path


def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para:
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs": [d]})
    return datum


def get_split_indices(data_sizes, chunk_sizes):
    ds = deepcopy(data_sizes)
    records = {}
    tmp = {}
    order = 0  # data_sizes index
    i = 0  # chunk_sizes index
    while len(data_sizes) > 0:
        d0 = data_sizes[0]
        c0 = chunk_sizes[0]
        if d0 > c0:
            val = c0 / ds[order]
        else:
            val = d0 / ds[order]

        if order not in tmp:
            rec = (0, val)
            tmp[order] = val
        else:
            rec = (tmp[order], tmp[order] + val)
            tmp[order] += val
        if i in records:
            records[i][order] = rec
        else:
            records[i] = {order: rec}

        if d0 > c0:
            data_sizes[0] -= c0
            chunk_sizes.pop(0)
            i += 1
        else:
            chunk_sizes[0] -= d0
            data_sizes.pop(0)
            order += 1
            if d0 == c0:
                chunk_sizes.pop(0)
                i += 1
    return records


def store_grad(get_ps, grads, grad_dims, task_id):
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[: i + 1])
            grads[beg:end, task_id].copy_(param.grad.data.view(-1))
        i += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
