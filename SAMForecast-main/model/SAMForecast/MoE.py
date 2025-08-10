from distutils.command.config import config
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from model.SAMForecast.LiftingWavelets import Model as model
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP

# Adaptive Multi-Scale Blocks
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_experts = configs.expert_nums
        self.output_size = configs.pred_len 
        self.input_size = configs.seq_len 
        self.k = configs.k

        self.start_linear = nn.Linear(in_features=configs.d_model, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()

        for wavelet_level in configs.wavelets_levels:
            self.experts.append(
                model(configs, wavelet_level)
            )

        if configs.features == 'M':
            self.w_gate = nn.Parameter(torch.zeros(configs.enc_in+4 , self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(configs.enc_in+4 , self.num_experts), requires_grad=True)
        elif configs.features == 'S':
            self.w_gate = nn.Parameter(torch.zeros(5 , self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(5 , self.num_experts), requires_grad=True)

        self.residual_connection = 1

        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        
        
        clean_values = torch.where(torch.isnan(clean_values), torch.zeros_like(clean_values), clean_values)
        
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        
        x = self.start_linear(x).squeeze(-1)
        clean_logits = x @ self.w_gate
        
        # clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, x_mark_enc, loss_coef=1e-2):
        # new_x = self.seasonality_and_trend_decompose(x)
        #multi-scale router
        # gates, load = self.noisy_top_k_gating(new_x, self.training)
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        x_enc_expert_inputs = dispatcher.dispatch(x)
        
        if(x_mark_enc!=None):
            x_mark_enc_expert_inputs = dispatcher.dispatch(x_mark_enc)
        else:
            x_mark_enc_expert_inputs = [None]*4
        expert_outputs = []
        regu_sum = 0
        for i in range(self.num_experts):
            b,_,_ = x_enc_expert_inputs[i].shape
            if(b!=0):
                expert_output, regu_sum = self.experts[i](x_enc_expert_inputs[i], x_mark_enc_expert_inputs[i])
                expert_outputs.append(expert_output)
                regu_sum +=regu_sum

        output = dispatcher.combine(expert_outputs)
        return output, balance_loss, regu_sum





