import torch.nn as nn


class DecoderConnector(nn.Module):
    def __init__(self, feature_dim, decoder_config):
        super().__init__()

        self.linear = nn.Linear(
            feature_dim,
            decoder_config.hidden_size * (1 + decoder_config.n_layer),
            bias=False,
        )

    def forward(self, feature):
        """
        Args:
            feature (Tensor): feature from Encoder or HippocampusModule,
                shape: (batch_size, feature_dim)
        Returns:
            decoder_inputs (Tensor): inputs to embedding and attention layers
                shape: (batch_size, (1 + n_layers) * n_embed)
        """

        decoder_inputs = self.linear(feature)

        return decoder_inputs


class Decoder(nn.Module):
    def __init__(self, transformer, feature_dim=None, tokens_weight=None):
        super().__init__()

        self.transformer = transformer

        if feature_dim is not None:
            self.feature_dim = feature_dim
            self.connector = DecoderConnector(feature_dim, transformer.config)

        self.loss_fct = nn.CrossEntropyLoss(weight=tokens_weight, ignore_index=-1)

        self.config = transformer.config

    def forward(self, input_ids, feature=None, past=None, labels=None):
        """
        Args:
            input_ids (LongTensor): input sequence of token ids
                shape: (batch_size, seq_len)
            feature (FloatTensor, optional): feature vectors of input text
                shape: (batch_size, feature_dim)
            past (FloatTensor): past inputs of each attention layer
                shape: (batch_size, seq_len, embed_dim)
            labels (LongTensor, optional): target prediction token ids.
                    note. label `-1` is ignore index.
                shape: (batch_size, seq_len)
        Returns:
            lm_logits (FloatTensor): logits for each vocabulary token
                shape: (batch_size, seq_len, vocab_size)
            past (FloatTensor): past inputs of each attention layer
                shape: (batch_size, num_layers, seq_len, seq_len)
            loss (FloatTensor): loss (output if labels is given)
                shape: (1,)
        """

        if feature is not None:
            feature = self.connector(feature)

        outputs = self.transformer(input_ids, latent=feature, past=past)

        lm_logits = outputs[0]

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs = (loss,) + outputs

        return outputs

    def resize_token_embeddings(self, len_tokenizer):
        self.transformer.resize_token_embeddings(len_tokenizer)
