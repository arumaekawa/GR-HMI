import logging
import os
import random

import torch
from encoder_decoder_model.models import CONFIG_NAME, Decoder, Encoder
from encoder_decoder_model.pretrain_settings import (
    DECODER_CLASS,
    DECODER_CONFIG,
    DECODER_SAVE_NAME,
    DECODER_SPECIAL_TOKENS,
    DECODER_TOKENIZER,
    ENCODER_CLASS,
    ENCODER_CONFIG,
    ENCODER_SAVE_NAME,
    ENCODER_SPECIAL_TOKENS,
    ENCODER_TOKENIZER,
    FILL_VAL,
    FINAL_DECODER_SAVE_NAME,
    FINAL_ENCODER_SAVE_NAME,
    args,
    init_logging,
)
from encoder_decoder_model.pretrain_utils import (
    PretrainDataset,
    TrainStep,
    WrapModel,
    create_dataloader,
    get_losses,
    get_raw_text_data,
    make_dir,
)
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelCriterion, DataParallelModel

# from torch.utils.data import DataLoader
from pytorch_transformers import AdamW
from scheduler import AnnealingLR
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


def train():

    # build model
    encoder_model = ENCODER_CLASS.from_pretrained(args.encoder_model_name)
    decoder_model = DECODER_CLASS.from_pretrained(
        args.decoder_model_name,
        config=DECODER_CONFIG,
        latent_as_gpt_emb=True,
        latent_as_gpt_attn=True,
    )
    encoder = Encoder(
        encoder_model, ENCODER_TOKENIZER, feature_dim=args.feature_dim
    ).cuda()
    decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda()

    # resize vocab size
    ENCODER_TOKENIZER.add_tokens(list(ENCODER_SPECIAL_TOKENS.values()))
    DECODER_TOKENIZER.add_tokens(list(DECODER_SPECIAL_TOKENS.values()))

    # save tokenizers
    if not args.only_lm:
        ENCODER_TOKENIZER.save_pretrained(
            os.path.join(args.pretrained_model_dir, "encoder")
        )
    DECODER_TOKENIZER.save_pretrained(
        os.path.join(args.pretrained_model_dir, "decoder")
    )

    # resize embedding
    encoder.resize_token_embeddings(len(ENCODER_TOKENIZER))
    decoder.resize_token_embeddings(len(DECODER_TOKENIZER))
    ENCODER_CONFIG.vocab_size = len(ENCODER_TOKENIZER)
    DECODER_CONFIG.vocab_size = len(DECODER_TOKENIZER)
    # save config
    if not args.only_lm:
        ENCODER_CONFIG.to_json_file(
            os.path.join(args.pretrained_model_dir, "encoder", CONFIG_NAME)
        )
    DECODER_CONFIG.to_json_file(
        os.path.join(args.pretrained_model_dir, "decoder", CONFIG_NAME)
    )

    # fp16 module
    if not args.fp32:
        encoder = FP16_Module(encoder)
        decoder = FP16_Module(decoder)

    # parallel model
    parallel_encoder = DataParallelModel(WrapModel(encoder), args.device_ids)
    parallel_decoder = DataParallelModel(WrapModel(decoder), args.device_ids)

    # dataset
    if args.train_data_path:
        args.train_data_path = os.path.join(
            args.pretrain_data_dir, args.train_data_path
        )
        train_data = PretrainDataset(data_path=args.train_data_path, data_type="train")
    else:
        raw_text_data = get_raw_text_data()
        random.shuffle(raw_text_data)
        train_text_data = raw_text_data[: int(len(raw_text_data) * 0.9)]
        # valid_text_data = raw_text_data[
        #     int(len(raw_text_data) * 0.9) : -int(len(raw_text_data) * 0.05)
        # ]
        # test_text_data = raw_text_data[-int(len(raw_text_data) * 0.05) :]
        train_data = PretrainDataset(train_text_data, data_type="train")
        # valid_data = PretrainDataset(valid_text_data, data_type="valid")
        # test_data = PretrainDataset(test_text_data, data_type="test")

    # dataloader
    max_train_batch_size = max(len(train_data) // args.min_n_steps, args.min_batch_size)
    train_loader = create_dataloader(train_data, "train", max_train_batch_size)
    n_train_optimization_steps = len(train_data) * args.n_train_epochs
    logger.info(
        "len of train dataset: {} , max train batch size {}".format(
            len(train_data), max_train_batch_size
        )
    )

    # trainable parameters
    param_optimizer = []
    param_optimizer += list(encoder.named_parameters())
    param_optimizer += list(decoder.named_parameters())

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

    # optimizer
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
    # learning rate scheduler
    scheduler = AnnealingLR(
        optimizer,
        start_lr=args.learning_rate,
        warmup_iter=int(args.n_warmup_ratio * len(train_data)),
        num_iters=int(n_train_optimization_steps),
        decay_style=args.decay_style,
    )

    # loss function
    train_loss_fct = DataParallelCriterion(
        CrossEntropyLoss(ignore_index=FILL_VAL), args.device_ids
    )

    tot_n_steps = 0
    train_once = TrainStep(encoder, decoder, optimizer, scheduler)

    encoder.train()
    decoder.train()
    for ep in range(args.n_train_epochs):
        cum_loss, cum_ae_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (enc_inputs, dec_inputs, dec_labels) in enumerate(train_loader):

            n_inputs = sum(len(_enc_inputs) for _enc_inputs in enc_inputs)

            # putting on each gpu device
            for i in range(len(enc_inputs)):
                enc_inputs[i] = (enc_inputs[i].to(args.device_ids[i]),)
                dec_inputs[i] = (dec_inputs[i].to(args.device_ids[i]),)
                dec_labels[i] = dec_labels[i].to(args.device_ids[i])

            # compute loss
            ae_loss, lm_loss, _ = get_losses(
                parallel_encoder,
                parallel_decoder,
                enc_inputs,
                dec_inputs,
                dec_labels,
                train_loss_fct,
            )
            loss = ae_loss + lm_loss
            del enc_inputs, dec_inputs, dec_labels
            # update parameters
            train_once(loss, n_inputs)

            # logging
            ae_loss = ae_loss.item() * n_inputs
            lm_loss = lm_loss.item() * n_inputs
            cum_loss += ae_loss + lm_loss
            cum_ae_loss += ae_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1) % args.logging_steps == 0:
                logger.info(
                    "progress {:.3f}, lr {:.1E}, loss {:.3f}, ae loss {:.3f}, "
                    "lm loss {:.3f}, avg batch size {:.1f}".format(
                        ep + cur_n_inputs / len(train_data),
                        scheduler.get_lr(),
                        cum_loss / cur_n_inputs,
                        cum_ae_loss / cur_n_inputs,
                        cum_lm_loss / cur_n_inputs,
                        cur_n_inputs / (n_steps + 1),
                    )
                )

        # save encoder and decoder at the end of each epoch
        if not args.only_lm:
            torch.save(
                encoder.state_dict(),
                os.path.join(
                    args.pretrained_model_dir,
                    "encoder",
                    ENCODER_SAVE_NAME + str(ep + 1),
                ),
            )
            torch.save(
                decoder.state_dict(),
                os.path.join(
                    args.pretrained_model_dir,
                    "decoder",
                    DECODER_SAVE_NAME + str(ep + 1),
                ),
            )
        else:
            torch.save(
                decoder.module.transformer.state_dict(),
                os.path.join(
                    args.pretrained_model_dir,
                    "decoder",
                    "decoder" + str(ep + 1),
                ),
            )

        # logging
        tot_n_steps += n_steps + 1
        logger.info(
            "epoch {}/{} done, tot steps {}, lr {:.1E}, loss {:.2f}, "
            "ae loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}".format(
                ep + 1,
                args.n_train_epochs,
                tot_n_steps,
                scheduler.get_lr(),
                cum_loss / cur_n_inputs,
                cum_ae_loss / cur_n_inputs,
                cum_lm_loss / cur_n_inputs,
                cur_n_inputs / (n_steps + 1),
            )
        )

    # save encoder and decoder at the end of last epoch
    if not args.only_lm:
        torch.save(
            encoder.state_dict(),
            os.path.join(args.pretrained_model_dir, "encoder", FINAL_ENCODER_SAVE_NAME),
        )
        torch.save(
            decoder.state_dict(),
            os.path.join(args.pretrained_model_dir, "decoder", FINAL_DECODER_SAVE_NAME),
        )
    else:
        torch.save(
            decoder.module.transformer.state_dict(),
            os.path.join(args.pretrained_model_dir, "decoder", "decoder-finish"),
        )

    return


if __name__ == "__main__":

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(
            logging.CRITICAL
        )

    # make save directory
    make_dir(args.pretrained_model_dir)
    if not args.only_lm:
        make_dir(os.path.join(args.pretrained_model_dir, "encoder"))
    make_dir(os.path.join(args.pretrained_model_dir, "decoder"))

    # initialize logger
    init_logging(os.path.join(args.pretrained_model_dir, "log_train.txt"))
    logger.info("args = {}".format(str(args)))

    # run training
    train()
