import torch
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from model.SAMForecast.MoE import Model as model
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums 
        self.num_nodes = configs.enc_in
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.num_experts_list = [4, 4, 4]
        self.d_model = configs.d_model
        self.revin = 1
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.enc_in, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                model(configs)
                    )
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )
        self.use_norm = configs.use_norm
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model,d_state=16,d_conv=2,expand=1) ,
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        balance_loss = 0
        # norm
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        regu_sum = 0
        out = enc_out
        for layer in self.AMS_lists:
            out, aux_loss, regu_sum = layer(out, x_mark_enc)
            balance_loss += aux_loss
            regu_sum +=regu_sum
            
        dec_out,attns = self.encoder(out, None, attn_mask=None)
        dec_out = self.projector(dec_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pre_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pre_len, 1))

        return dec_out, balance_loss, regu_sum,

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, balance_loss, regu_sum, = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pre_len:, :], regu_sum, balance_loss  # [B, L, D]
