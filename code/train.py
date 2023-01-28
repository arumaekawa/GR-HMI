import logging
import os

import torch
from encoder_decoder_model.models import Decoder, Encoder
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelCriterion, DataParallelModel
from pytorch_transformers import AdamW
from regularizers import REG_TYPE_KEYS, REG_TYPES, Weight_Regularized_AdamW
from scheduler import AnnealingLR
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
    FILL_VAL,
    FINAL_DECODER_SAVE_NAME,
    FINAL_ENCODER_SAVE_NAME,
    HIPPOCAMPUS_MODULE_CLASS,
    LLL_METHODS,
    TASK_DICT,
    args,
    init_logging,
)

# from regularizers import Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
from utils import GEMStep  # get_ae_loss,; get_lm_loss,; get_qa_loss,
from utils import (
    QADataset,
    TrainStep,
    WrapModel,
    add_all_feature_to_hc_memory,
    create_dataloader,
    create_extra_data,
    get_gen_token,
    get_hc_module_memory,
    get_loss_of_all_train_data,
    get_losses,
    get_model_dir,
    get_real_data,
    lll_unbound_setting,
    make_dir,
)

logger = logging.getLogger(__name__)


def train(task_ids, encoder, decoder, hc_module):
    # current tasks
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info(
        "start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type)
    )
    # make current task directory
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)
    make_dir(os.path.join(model_dir, "decoder"))
    if args.seq_train_type == "hmi-lamol":
        make_dir(os.path.join(model_dir, "encoder"))
        make_dir(os.path.join(model_dir, "hippocampus_module"))

    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    # retrieve or generate replay samples of previous tasks
    train_extra_data = []
    if task_ids[0] > 0:
        if args.seq_train_type in LLL_METHODS and not args.skip_tasks:
            prev_task = args.tasks[task_ids[0] - 1]
            with torch.no_grad():
                # retrieve features of previous task samples from hc_module
                if args.seq_train_type == "hmi-lamol":
                    hc_memory = get_hc_module_memory(hc_module, tasks[0])
                else:
                    hc_memory = None
                # decode replay samples from features
                create_extra_data(
                    tasks[0],
                    prev_task,
                    decoder,
                    train_extra_data,
                    hc_memory=hc_memory,
                )
        elif "gem" in args.seq_train_type:
            get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
            args.memory_data.append(train_extra_data)
            train_extra_data = []
    logger.info("extra training data size: {}".format(len(train_extra_data)))

    if not decoder:
        if args.seq_train_type == "hmi-lamol":
            if args.pretrained_model_dir is not None:
                # build pretrained language model
                encoder_model = ENCODER_CLASS(ENCODER_CONFIG)
                decoder_model = DECODER_CLASS(
                    DECODER_CONFIG, latent_as_gpt_emb=True, latent_as_gpt_attn=True
                )
                # build encoder and decoder model on gpu device
                encoder = Encoder(
                    encoder_model, ENCODER_TOKENIZER, feature_dim=args.feature_dim
                ).cuda()
                decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda()
                # load pretrained parameters
                logger.info(
                    "Load pretrained encoder-decoder model from {}".format(
                        args.pretrained_model_dir
                    )
                )
                encoder.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.pretrained_model_dir, "encoder", "encoder-finish"
                        )
                    )
                )
                decoder.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.pretrained_model_dir, "decoder", "decoder-finish"
                        )
                    )
                )
            else:
                # load pretrained language model
                encoder_model = ENCODER_CLASS.from_pretrained(args.encoder_model_name)
                decoder_model = DECODER_CLASS.from_pretrained(
                    args.decoder_model_name,
                    config=DECODER_CONFIG,
                    latent_as_gpt_emb=True,
                    latent_as_gpt_attn=True,
                )
                # build encoder and decoder model on gpu device
                encoder = Encoder(
                    encoder_model, ENCODER_TOKENIZER, feature_dim=args.feature_dim
                ).cuda()
                decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda()
            # hippocampus module
            hc_module = HIPPOCAMPUS_MODULE_CLASS(
                args.feature_dim, fp32=args.fp32, max_memory_size=args.max_memory_size
            )
        else:
            # load pretrained language model
            if args.pretrained_model_dir is not None:
                # load pretrained decoder
                decoder_model = DECODER_CLASS(config=DECODER_CONFIG)
                decoder_model.load_state_dict(
                    torch.load(
                        os.path.join(
                            args.pretrained_model_dir, "decoder", "decoder-finish"
                        )
                    )
                )
                decoder = Decoder(decoder_model).cuda()
            else:
                # load pretrained language model
                decoder_model = DECODER_CLASS.from_pretrained(args.decoder_model_name)
                # build decoder model
                decoder = Decoder(decoder_model).cuda()

        # resize token embeddings
        if args.seq_train_type == "hmi-lamol":
            encoder.resize_token_embeddings(len(ENCODER_TOKENIZER))
        decoder.resize_token_embeddings(len(DECODER_TOKENIZER))

    # add gen_token to decoder tokenizer
    gen_token = get_gen_token(tasks[0])
    DECODER_TOKENIZER.add_tokens([gen_token])
    DECODER_TOKENIZER.save_pretrained(model_dir)
    DECODER_SPECIAL_TOKENS[tasks[0]] = gen_token
    DECODER_SPECIAL_TOKEN_IDS[tasks[0]] = DECODER_TOKENIZER.convert_tokens_to_ids(
        gen_token
    )
    # logging info about gen token
    logger.info(
        "gen token = {} , gen token id = {}".format(
            gen_token, DECODER_SPECIAL_TOKEN_IDS[tasks[0]]
        )
    )
    # update config
    if args.seq_train_type == "hmi-lamol":
        ENCODER_CONFIG.vocab_size = len(ENCODER_TOKENIZER)
        ENCODER_CONFIG.to_json_file(
            os.path.join(model_dir, "encoder", ENCODER_SAVE_NAME + CONFIG_NAME)
        )
    DECODER_CONFIG.vocab_size = len(DECODER_TOKENIZER)
    DECODER_CONFIG.to_json_file(
        os.path.join(model_dir, "decoder", DECODER_SAVE_NAME + CONFIG_NAME)
    )
    global TOKENS_WEIGHT
    if len(DECODER_TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        raise NotImplementedError
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_DECODER_SAVE_NAME)
                config_path = os.path.join(model_dir, CONFIG_NAME)
                decoder_config = DECODER_CONFIG_CLASS.from_json_file(config_path)
                decoder_model = DECODER_CLASS(decoder_config)
                decoder = Decoder(decoder_model).cuda()
                state_dict = torch.load(model_path)
                decoder.load_state_dict(state_dict)
                if not args.fp32:
                    decoder = FP16_Module(decoder)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calculating reg_params ...")
                    train_qadata = QADataset(
                        train_dataset,
                        "train",
                        DECODER_SPECIAL_TOKEN_IDS[tasks[0]],
                        train_extra_data,
                    )
                    max_train_batch_size = max(
                        len(train_qadata) // args.min_n_steps, args.min_batch_size
                    )
                    train_dataloader = create_dataloader(
                        train_qadata, "train", max_train_batch_size
                    )
                    parallel_model = DataParallelModel(
                        WrapModel(decoder), args.device_ids
                    )
                    regularizer = REG_TYPES[args.seq_train_type](
                        decoder, parallel_model, [train_dataloader], tasks[0]
                    )
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(
                        decoder.state_dict(),
                        os.path.join(model_dir, FINAL_DECODER_SAVE_NAME),
                    )
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            return decoder

    # resize token embeddings because added gen_token
    decoder.resize_token_embeddings(len(DECODER_TOKENIZER))

    # to fp16 module
    if not args.fp32:
        if args.seq_train_type == "hmi-lamol":
            encoder = FP16_Module(encoder)
        decoder = FP16_Module(decoder)

    # parallel model
    if args.seq_train_type == "hmi-lamol":
        parallel_encoder = DataParallelModel(WrapModel(encoder), args.device_ids)
    else:
        parallel_encoder = None
    parallel_decoder = DataParallelModel(WrapModel(decoder), args.device_ids)

    # train dataset as QA dataset
    train_qadata = QADataset(
        train_dataset, "train", DECODER_SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data
    )
    # max train batch size
    max_train_batch_size = max(
        len(train_qadata) // args.min_n_steps, args.min_batch_size
    )
    # train dataloader
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    # training epochs
    if not args.unbound and args.seq_train_type != "multitask":
        # n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs["_".join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    # logging information about number of training steps
    logger.info(
        "len of train dataset: {}, max train batch size {}, "
        "num of opt steps: {}".format(
            len(train_qadata), max_train_batch_size, n_train_optimization_steps
        )
    )

    # parameters to optimize
    param_optimizer = list(decoder.named_parameters())
    if args.seq_train_type == "hmi-lamol":
        if args.train_encoder:
            param_optimizer += list(encoder.named_parameters())
        else:
            for p in encoder.parameters():
                p.requires_grad = False

    # parameters to or not to apply weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if "gem" in args.seq_train_type:
        decoder.task_id = task_ids[0]
        if not hasattr(decoder, "grad_dims"):
            decoder.grad_dims = []
            for param in decoder.parameters():
                decoder.grad_dims.append(param.data.numel())
        if not hasattr(decoder, "grads"):
            decoder.grads = torch.zeros(sum(decoder.grad_dims), len(args.tasks))
            decoder.grads = decoder.grads.cuda()

    # optimizer
    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    if not args.fp32:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=None,
            dynamic_loss_scale=True,
            dynamic_loss_args={"scale_window": 100, "min_scale": 1, "delayed_shift": 2},
        )

    # scheduler
    scheduler = AnnealingLR(
        optimizer,
        start_lr=args.learning_rate,
        warmup_iter=int(args.n_warmup_ratio * len(train_qadata)),
        num_iters=int(n_train_optimization_steps),
        decay_style=args.decay_style,
    )
    # loss function
    train_loss_fct = DataParallelCriterion(
        CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids
    )

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(
            train_qadata, "train", max_train_batch_size
        )
        prev_task = args.tasks[task_ids[0] - 1]
        regularizer = REG_TYPES[args.seq_train_type](
            decoder, parallel_decoder, [copy_train_dataloader], tasks[0], prev_task
        )
        regularizer.task_start_do()

    # reset new tasks memory of hippocampus module
    if hc_module is not None:
        hc_module.reset_new_task(tasks[0])
        add_all_feature_to_hc_memory(train_qadata, parallel_encoder, hc_module)
        if "pq" in args.hc_module_type:
            logger.info("Quantize hippocampus module memory")
            hc_module.quantize_memory()

        if "loss_diff" in args.hc_module_type:
            logger.info("Add all sample loss before training")
            before_losses = get_loss_of_all_train_data(
                train_qadata,
                parallel_encoder,
                parallel_decoder,
                train_loss_fct,
                hc_module,
                loss_type="both",
            )
            hc_module.add_before_loss(before_losses)

    tot_n_steps = 0
    train_once = TrainStep(decoder, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(decoder, parallel_decoder, train_loss_fct, optimizer)
    # model to training mode
    decoder.train()
    if args.seq_train_type == "hmi-lamol":
        encoder.train() if args.train_encoder else encoder.eval()
    # training loop
    for ep in range(n_train_epochs):
        cum_loss, cum_qa_loss, cum_gen_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (cls_cq, _, _, cqa, _, qa_Y, gen_Y, is_replay) in enumerate(
            train_dataloader
        ):

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)

            for i in range(len(cqa)):
                if cls_cq is not None:
                    cls_cq[i] = (cls_cq[i].to(args.device_ids[i]),)
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                qa_Y[i] = qa_Y[i].to(args.device_ids[i])
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])

            # task loss (as question answering task)
            qa_loss, gen_loss = get_losses(
                parallel_decoder,
                cqa,
                qa_Y,
                gen_Y,
                train_loss_fct,
                parallel_encoder=parallel_encoder,
                cls_cq=cls_cq,
                hc_module=hc_module,
            )

            # total loss
            loss = qa_loss + args.gen_lambda * gen_loss

            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])

            # update model with total loss
            train_once(loss, n_inputs)

            # logging
            qa_loss = qa_loss.item() * n_inputs
            gen_loss = args.gen_lambda * gen_loss.item() * n_inputs
            cum_loss += qa_loss + gen_loss
            cum_qa_loss += qa_loss
            cum_gen_loss += gen_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1) % args.logging_steps == 0:
                logger.info(
                    "progress {:.3f}, lr {:.1E}, loss {:.3f}, qa loss {:.3f}, "
                    "gen loss {:.3f}, avg batch size {:.1f}".format(
                        ep + cur_n_inputs / len(train_qadata),
                        scheduler.get_lr(),
                        cum_loss / cur_n_inputs,
                        cum_qa_loss / cur_n_inputs,
                        cum_gen_loss / cur_n_inputs,
                        cur_n_inputs / (n_steps + 1),
                    )
                )

        # save models at the end of each epoch
        if args.seq_train_type == "hmi-lamol" and args.train_encoder:
            logger.info(
                "Save encoder in `{}`".format(
                    os.path.join(model_dir, "encoder", ENCODER_SAVE_NAME + str(ep + 1))
                )
            )
            torch.save(
                encoder.state_dict(),
                os.path.join(model_dir, "encoder", ENCODER_SAVE_NAME + str(ep + 1)),
            )
        logger.info(
            "Save decoder in `{}`".format(
                os.path.join(model_dir, "decoder", DECODER_SAVE_NAME + str(ep + 1))
            )
        )
        torch.save(
            decoder.state_dict(),
            os.path.join(model_dir, "decoder", DECODER_SAVE_NAME + str(ep + 1)),
        )
        # logging
        tot_n_steps += n_steps + 1
        logger.info(
            "epoch {}/{} done, tot steps {}, lr {:.1E}, loss {:.2f}, "
            "qa loss {:.2f}, gen loss {:.2f}, avg batch size {:.1f}".format(
                ep + 1,
                n_train_epochs,
                tot_n_steps,
                scheduler.get_lr(),
                cum_loss / cur_n_inputs,
                cum_qa_loss / cur_n_inputs,
                cum_gen_loss / cur_n_inputs,
                cur_n_inputs / (n_steps + 1),
            )
        )

    if hc_module is not None:
        if "loss_diff" in args.hc_module_type:
            logger.info("Add all sample loss after training")
            after_losses = get_loss_of_all_train_data(
                train_qadata,
                parallel_encoder,
                parallel_decoder,
                train_loss_fct,
                hc_module,
                loss_type="both",
            )
            hc_module.add_after_loss(after_losses)
        elif "low_ppl" in args.hc_module_type:
            logger.info("Add all sample gen loss after training")
            gen_losses = get_loss_of_all_train_data(
                train_qadata,
                parallel_encoder,
                parallel_decoder,
                train_loss_fct,
                hc_module,
                loss_type="gen",
            )
            hc_module.add_gen_loss(gen_losses)

        if args.max_memory_size is not None:
            logger.info("Pruning hippocampus module memory")
            hc_module.prune_memory()
        logger.info(f"memory size (numbers): {hc_module.memory_size}")
        logger.info(f"memory size (bytes): {hc_module.memory_byte_size}")
        hc_module.save_pretrained(os.path.join(model_dir, "hippocampus_module"))

    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()

    # save mode
    if args.seq_train_type == "hmi-lamol" and args.train_encoder:
        logger.info(
            "Save encoder in `{}`".format(
                os.path.join(model_dir, "encoder", FINAL_ENCODER_SAVE_NAME)
            )
        )
        torch.save(
            encoder.state_dict(),
            os.path.join(model_dir, "encoder", FINAL_ENCODER_SAVE_NAME),
        )
    logger.info(
        "Save decoder in `{}`".format(
            os.path.join(model_dir, "decoder", FINAL_DECODER_SAVE_NAME)
        )
    )
    torch.save(
        decoder.state_dict(),
        os.path.join(model_dir, "decoder", FINAL_DECODER_SAVE_NAME),
    )

    return encoder, decoder, hc_module


if __name__ == "__main__":

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(
            logging.CRITICAL
        )

    # make save outputs directory
    assert not os.path.exists(args.model_dir_root), args.model_dir_root
    make_dir(args.model_dir_root)

    # initialize logger
    init_logging(os.path.join(args.model_dir_root, "log_train.txt"))
    logger.info("args = {}".format(str(args)))

    # run continual learning
    encoder, decoder, hc_module = None, None, None
    if args.seq_train_type == "multitask":
        decoder = train(list(range(len(args.tasks))), encoder, decoder, hc_module)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        for task_id in range(len(args.tasks)):
            encoder, decoder, hc_module = train([task_id], encoder, decoder, hc_module)
