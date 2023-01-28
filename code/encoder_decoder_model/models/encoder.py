import torch
import torch.nn as nn


class EncoderConnector(nn.Module):
    def __init__(self, feature_dim, encoder_config):
        super().__init__()

        self.linear = nn.Linear(encoder_config.hidden_size, feature_dim, bias=False)

    def forward(self, cls_hidden_state):
        """
        Args:
            cls_hidden_state (Tensor): Encoder last hidden state of CLS token,
                shape: (batch_size, n_embed)
        Returns
            feature (Tensor): feature vectors of input text
                shape: (batch_size, feature_dim)
        """

        feature = self.linear(cls_hidden_state)

        return feature


class Encoder(nn.Module):
    def __init__(self, transformer, tokenizer, feature_dim, activation=torch.tanh):
        super().__init__()

        self.feature_dim = feature_dim

        self.transformer = transformer
        self.connector = EncoderConnector(self.feature_dim, self.transformer.config)

        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.config = transformer.config

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): the sequence to the encoder
                shape: (batch_size, seq_len)
            attention_mask (Tensor, optional):
                mask for attention layers, (default: mask pad_tokens)
                    - `1` for tokens that are `not masked`,
                    - `0` for tokens that are `masked`.
                shape: (batch_size, seq_len)
        Return:
            feature (Tensor): representation vector of the input sequence
                shape: (batch_size, feature_dim)
            last_hidden_state (Tensor): representation for all input tokens
                shape: (batch_size, seq_len, embed_dim)
        """

        assert all(input_ids[:, 0].eq(self.cls_token_id))

        if attention_mask is None:
            attention_mask = input_ids.eq(self.pad_token_id).logical_not().float()

        last_hidden_state, cls_pooled_hidden_state = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )[:2]

        feature = self.connector(cls_pooled_hidden_state)

        return feature, last_hidden_state

    def resize_token_embeddings(self, len_tokenizer):
        self.transformer.resize_token_embeddings(len_tokenizer)
