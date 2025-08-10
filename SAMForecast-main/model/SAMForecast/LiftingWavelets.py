import torch
import torch.nn as nn
from layers.lifting import LiftingScheme

class Model(nn.Module):
    def __init__(self, configs, wavelet_level):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = nn.Dropout(p=configs.dropout)
        self.levels = nn.ModuleList()
        self.ps = nn.ModuleList()
        self.level_num = wavelet_level
        self.projector = nn.Linear(configs.d_model+wavelet_level+1, configs.d_model, bias=True)
        for i in range(wavelet_level):
            if configs.features == 'M':
                self.levels.add_module(
                    'level_' + str(i),
                    LevelTWaveNet(configs.enc_in+4,3,0.01,0.01)
                )
            elif configs.features == 'S':
                self.levels.add_module(
                    'level_' + str(i),
                        LevelTWaveNet(5,3,0.01,0.01)
                    )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forecast(self, x_enc, x_mark_enc):
        enc_out = x_enc 
        regu_sum = [] 
        approx = self.avgpool(enc_out)
        coefs = [approx]
        l_input = enc_out
        
        for l in self.levels:
            l_input, details, regu = l(l_input)
            regu_sum += [regu]
            de_avg = self.avgpool(details)
            coefs = coefs + [de_avg]
        coefs = torch.cat(coefs, 2)
        coefs = torch.cat((coefs,enc_out),2)
        dec_out = self.projector(coefs)
        return dec_out, regu_sum
        

    def forward(self, x_enc, x_mark_enc, mask=None):
        dec_out, regu_sum = self.forecast(x_enc, x_mark_enc)
        return dec_out, regu_sum  # [B, L, D]

 
class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        """
        Conduct decomposition and calculate regularization terms
        :param x: [batch_size, in_planes, sequence_length]。
        :return: approx component, details component, regularization terms
        """
        global regu_d, regu_c
        (L, H) = self.wavelet(x)  
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            # Constrain on the approximation
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                # Only the details
                regu = regu_d
            elif self.regu_details == 0.0:
                # Only the approximation
                regu = regu_c
            else:
                # Both
                regu = regu_d + regu_c

            return approx, details, regu

    