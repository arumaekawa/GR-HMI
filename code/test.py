import csv
import json
import logging
import os
from collections import OrderedDict

import torch
from encoder_decoder_model.models import Decoder, Encoder
from fp16 import FP16_Module
from metrics import compute_metrics
from settings import (
    CONFIG_NAME,
    DECODER_CLASS,
    DECODER_CONFIG,
    DECODER_CONFIG_CLASS,
    DECODER_SAVE_NAME,
    DECODER_SPECIAL_TOKEN_IDS,
    DECODER_SPECIAL_TOKENS,
    DECODER_TOKENIZER,
    ENCODER_CLASS,
    ENCODER_CONFIG,
    ENCODER_SAVE_NAME,
    ENCODER_TOKENIZER,
    HIPPOCAMPUS_MODULE_CLASS,
    TASK_DICT,
    args,
    init_logging,
)
from utils import (
    QADataset,
    create_dataloader,
    get_gen_token,
    get_model_dir,
    lll_unbound_setting,
    logits_to_tokens,
    sample_sequence,
)

logger = logging.getLogger(__name__)


def test_one_to_one(
    task_load, task_eval, decoder, score_dict, encoder=None, hc_module=None
):
    logger.info(
        "start to test { task: %s (load) %s (eval), seq train type: %s }"
        % (task_load, task_eval, args.seq_train_type)
    )

    test_qadata = QADataset(
        TASK_DICT[task_eval]["test"], "test", DECODER_SPECIAL_TOKEN_IDS[task_load]
    ).sort()
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(DECODER_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]
    if args.seq_train_type == "hmi-lamol":
        features = [0 for _ in range(n_examples)]
    else:
        features = None

    cnt = 0
    for n_steps, (cls_cqs, cqs, len_cqs, _, _, _, _, _) in enumerate(test_dataloader):
        # assume n_gpus == 1
        if cls_cqs is not None:
            cls_cqs = cls_cqs[0]
        cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = cqs.shape[0]
        if args.seq_train_type == "hmi-lamol":
            assert encoder is not None and hc_module is not None
            feature = encoder(cls_cqs.cuda())[0]
            if not args.fp32:
                feature = feature.half()
            if "pq" in args.hc_module_type:
                feature = hc_module.reform_feature(feature.cpu()).cuda()
        else:
            feature = None

        all_outputs = decoder(input_ids=cqs.cuda(), feature=feature)
        outputs = all_outputs[0]
        if args.decoder_model_name == "gpt2":
            pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), len_cqs - 1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt]["dec_examples"]["len_cq"]
            qa_results[cnt] = cqs[i][: len_cqs[i]]
            if next_tokens[i] != DECODER_SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((cqs[i][: len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) < min(max_tot_lens[cnt], args.decoder_max_len):
                    need_process.update([[cnt, None]])
                    if args.decoder_model_name == "gpt2":
                        for layer_id in range(DECODER_CONFIG.n_layer):
                            all_pasts[layer_id][cnt] = pasts[layer_id][
                                :, i, ..., : len_cqs[i], :
                            ].type(torch.float32 if args.fp32 else torch.half)
            if args.seq_train_type == "hmi-lamol":
                features[cnt] = feature[i]
            cnt += 1

        # dynamic threshold to avoid out of memory
        if len(need_process) > int(12 * args.memory_sizes[0] / cqs.shape[1]):
            sample_sequence(
                decoder,
                need_process,
                qa_results,
                all_pasts,
                max_tot_lens,
                features=features,
                top_k=args.top_k_qa,
                top_p=args.top_p_qa,
            )
    sample_sequence(
        decoder,
        need_process,
        qa_results,
        all_pasts,
        max_tot_lens,
        features=features,
        top_k=args.top_k_qa,
        top_p=args.top_p_qa,
    )

    if task_eval in ["wikisql", "woz.en", "multinli.in.out"]:
        ids = test_qadata.get_indices()
        test_qadata.sort_by_index()
        qa_results = [x[1] for x in sorted([(i, g) for i, g in zip(ids, qa_results)])]
    for i in range(len(test_qadata)):
        _, len_cq, _, _, qa_Y, _, = test_qadata[
            i
        ]["dec_examples"].values()
        if task_eval in ["wikisql", "woz.en"]:
            qa_Y = test_qadata.answers[i]
        else:
            qa_Y = list(filter(lambda x: x != -1, qa_Y))[:-1]  # remove eos
            qa_Y = " ".join([str(y) for y in qa_Y]).split(
                str(DECODER_SPECIAL_TOKEN_IDS["pad_token"])
            )
            qa_Y = [DECODER_TOKENIZER.decode(list(map(int, y.split()))) for y in qa_Y]
        qa_results[i] = [
            DECODER_TOKENIZER.decode(qa_results[i].tolist()[len_cq:]),
            qa_Y,
        ]
    get_test_score(task_eval, qa_results, score_dict)

    model_dir = decoder.model_dir
    ep = decoder.ep
    results_path = os.path.join(model_dir, "qa_{}_{}.csv".format(task_eval, ep + 1))
    if not args.debug:
        with open(results_path, "w", encoding="utf-8") as f:
            qa_writer = csv.writer(f, delimiter=",")
            qa_writer.writerow(["y", "pred"])
            for pred, y in qa_results:
                if task_eval == "wikisql":
                    y = y["answer"]
                elif task_eval == "woz.en":
                    y = y[1]
                qa_writer.writerow([y, pred])

    return decoder, score_dict


def get_test_score(task_eval, qa_results, score_dict):

    score = compute_metrics(
        qa_results,
        bleu="iwslt.en.de" in task_eval or "multinli.in.out" in task_eval,
        dialogue="woz.en" in task_eval,
        rouge="cnn_dailymail" in task_eval,
        logical_form="wikisql" in task_eval,
        corpus_f1="zre" in task_eval,
    )
    score_dict[task_eval] = score


def test_one_to_many(task_load):
    score_dicts = []
    for ep in range(args.n_train_epochs[task_load]):
        model_dir = get_model_dir([task_load])
        # encoder path
        if args.seq_train_type == "hmi-lamol":
            if args.train_encoder:
                encoder_path = os.path.join(
                    model_dir, "encoder", ENCODER_SAVE_NAME + str(ep + 1)
                )
            else:
                encoder_path = os.path.join(
                    args.pretrained_model_dir, "encoder", "encoder-finish"
                )
        # decoder path
        decoder_path = os.path.join(
            model_dir, "decoder", DECODER_SAVE_NAME + str(ep + 1)
        )
        config_path = os.path.join(
            model_dir, "decoder", DECODER_SAVE_NAME + CONFIG_NAME
        )

        gen_token = get_gen_token(task_load)
        DECODER_TOKENIZER.add_tokens([gen_token])
        DECODER_SPECIAL_TOKENS[task_load] = gen_token
        DECODER_SPECIAL_TOKEN_IDS[task_load] = DECODER_TOKENIZER.convert_tokens_to_ids(
            gen_token
        )
        decoder_config = DECODER_CONFIG_CLASS.from_json_file(config_path)
        if args.seq_train_type == "hmi-lamol":
            # encoder
            encoder_model = ENCODER_CLASS(ENCODER_CONFIG)
            encoder = (
                Encoder(encoder_model, ENCODER_TOKENIZER, feature_dim=args.feature_dim)
                .cuda()
                .eval()
            )
            encoder.load_state_dict(torch.load(encoder_path))
            # decoder
            decoder_model = DECODER_CLASS(
                decoder_config, latent_as_gpt_emb=True, latent_as_gpt_attn=True
            )
            decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda().eval()
            # hippocampus module
            hc_module = HIPPOCAMPUS_MODULE_CLASS(
                args.feature_dim, fp32=args.fp32, max_memory_size=args.max_memory_size
            )
            hc_module.from_pretrained(os.path.join(model_dir, "hippocampus_module"))
        else:
            # encoder
            encoder = None
            # decoder
            decoder_model = DECODER_CLASS(decoder_config)
            decoder = Decoder(decoder_model).cuda().eval()
            # hippocampus module
            hc_module = None

        decoder_state_dict = torch.load(decoder_path, map_location="cuda:0")
        decoder.load_state_dict(decoder_state_dict)

        if not args.fp32:
            decoder = FP16_Module(decoder)

        decoder.ep = ep
        decoder.model_dir = model_dir
        logger.info("task: {}, epoch: {}".format(task_load, ep + 1))
        score_dict = {k: None for k in args.tasks}
        with torch.no_grad():
            for task_eval in args.tasks:
                test_one_to_one(
                    task_load,
                    task_eval,
                    decoder,
                    score_dict,
                    encoder=encoder,
                    hc_module=hc_module,
                )
        logger.info("score: {}".format(score_dict))
        score_dicts.append(score_dict)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(score_dicts, f)


if __name__ == "__main__":
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")

    if args.decoder_model_name == "gpt2":
        args.fp32 = False  # always use fp16 in testing

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(
            logging.CRITICAL
        )
    # initialize logger
    init_logging(os.path.join(args.model_dir_root, "log_test.txt"))
    logger.info("args = {}".format(args))

    # run test
    if args.seq_train_type == "multitask":
        test_one_to_many("_".join(args.tasks))
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(
                split_size=args.unbound, data_type="test", test_target="origin"
            )
            for task_load in args.splitted_tasks:
                test_one_to_many(task_load)
        else:
            for task_load in args.tasks:
                test_one_to_many(task_load)
