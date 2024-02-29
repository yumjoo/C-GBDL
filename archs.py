import torch
import random
import math
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from base.correlation import build_corr_network
from torch.distributions import MultivariateNormal



position_embedding = 'sine'
hidden_dim = 64
dropout = 0.1
nheads = 4
dim_feedforward = 128
featurefusion_layers = 1
activation = 'relu'
alpha = 0.2

depth = 32
height = 128
weight = 128

__all__ = ['MC_UNet', 'LRL']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out


class VGGBlock_MC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x, train=True):
        out = self.conv1(F.dropout3d(x, training=train, p=0.3))
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out


class MC_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 96, 192, 384]

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock_MC(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock_MC(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock_MC(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock_MC(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, train=True, T=1):
        
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        output_final = None

        for i in range(0, T):

            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

            output = self.final(x0_4)

            if T <= 1:
                output_final = output
            else:
                if i == 0:
                    output_final = output.unsqueeze(0)
                else:
                    output_final = torch.cat((output_final, output.unsqueeze(0)), dim=0)

        return output_final


class LRL(nn.Module):
    def __init__(self, num_classes, input_channels=3, resize_w=128, resize_h=128, **kwargs):
        super().__init__()

        self.nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

        w = int(resize_w / 32)
        h = int(resize_h / 32)

        self.conv0_x = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv1_x = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv2_x = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv3_x = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv4_x = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4], nn.ReLU(inplace=True))

        self.conv_down = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_down_ = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.mean_vec = nn.Linear(self.nb_filter[4] * w * h, int(self.nb_filter[3] / 2))

        self.covar_half = nn.Linear(self.nb_filter[4] * w * h, int(self.nb_filter[3] / 2))
        self.covar_diag = nn.Linear(self.nb_filter[4] * w * h, int(self.nb_filter[3] / 2))

        self.x_ori_vec = nn.Linear(self.nb_filter[4] * w * h, int(self.nb_filter[3] / 2))

        self.transform = nn.Linear(int(self.nb_filter[3] / 2), self.nb_filter[4] * w * h)
        self.transform_ = nn.Linear(int(self.nb_filter[3] / 2), self.nb_filter[4] * w * h)

        self.conv_up = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.conv_up_ = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2),
                                           padding=(0, 1, 1))

        self.conv3_1_x = VGGBlock(self.nb_filter[3] + self.nb_filter[4], self.nb_filter[3], self.nb_filter[3],
                                  nn.ReLU(inplace=True))
        self.conv2_2_x = VGGBlock(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2], self.nb_filter[2],
                                  nn.ReLU(inplace=True))
        self.conv1_3_x = VGGBlock(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1], self.nb_filter[1],
                                  nn.ReLU(inplace=True))
        self.conv0_4_x = VGGBlock(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0],
                                  nn.ReLU(inplace=True))

        self.conv00 = nn.Conv3d(self.nb_filter[0], input_channels, 3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        self.conv0_0 = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv0_0_ = VGGBlock(input_channels+1, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv4_0 = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4], nn.ReLU(inplace=True))

        self.conv3_1 = VGGBlock(self.nb_filter[3] + self.nb_filter[4] + self.nb_filter[4], self.nb_filter[3],
                                self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv2_2 = VGGBlock(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2], self.nb_filter[2],
                                nn.ReLU(inplace=True))
        self.conv1_3 = VGGBlock(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1], self.nb_filter[1],
                                nn.ReLU(inplace=True))
        self.conv0_4 = VGGBlock(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0],
                                nn.ReLU(inplace=True))

        self.final = nn.Conv3d(self.nb_filter[0], num_classes, kernel_size=1)

        # ----------------------------

        self.conv_down_32 = nn.Sequential(
            nn.Conv3d(self.nb_filter[2], self.nb_filter[2], (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.InstanceNorm3d(self.nb_filter[2]),
            nn.ReLU(inplace=True)
        )

        self.conv_down_16 = nn.Sequential(
            nn.Conv3d(self.nb_filter[3], self.nb_filter[3], (2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.InstanceNorm3d(self.nb_filter[3]),
            nn.ReLU(inplace=True)
        )

        self.atten_32_ = build_corr_network(self.nb_filter[2], dropout, nheads,
                                            self.nb_filter[2] * 2, featurefusion_layers, activation)

        self.atten_16_ = build_corr_network(self.nb_filter[3], dropout, nheads,
                                            self.nb_filter[3] * 2, featurefusion_layers, activation)

        self.conv_up_32 = nn.ConvTranspose3d(self.nb_filter[2], self.nb_filter[2], (2, 3, 3), stride=(2, 1, 1),
                                             padding=(0, 1, 1))
        self.in_32 = nn.InstanceNorm3d(self.nb_filter[2])
        self.relu_32 = nn.ReLU(inplace=True)

        self.conv_up_16 = nn.ConvTranspose3d(self.nb_filter[3], self.nb_filter[3], (2, 3, 3), stride=(2, 1, 1),
                                             padding=(0, 1, 1))
        self.in_16 = nn.InstanceNorm3d(self.nb_filter[3])
        self.relu_16 = nn.ReLU(inplace=True)

        self.avg_pool_32 = nn.AvgPool3d(kernel_size=(1, height // 4, weight // 4))
        self.avg_pool_16 = nn.AvgPool3d(kernel_size=(1, height // 8, weight // 8))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def mask_feature(self, features, fixed_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(fixed_mask.float(), feature.size()[2:], mode='trilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def cosine_similarity(self, feature_map1, feature_map2):
        feature_map1 = feature_map1.view(feature_map1.size(0), -1)
        feature_map2 = feature_map2.view(feature_map2.size(0), -1)

        feature1 = F.normalize(feature_map1)
        feature2 = F.normalize(feature_map2)

        return feature1.mm(feature2.t())

    def uncertainty_map(self, prob_map):
        entropy = -torch.sum(prob_map * torch.log2(prob_map + 1e-10), dim=1)
        return entropy

    def reparameterize(self, mu, covar_half, M=1):

        B, depth, D = mu.size()

        if M == 1:
            eps = torch.randn(B * depth, 1, D).cuda()
            eps_ = torch.bmm(eps, covar_half)
            return eps_.view(B, depth, D) + mu
        else:
            latent = None
            # latent will be (M, B, depth, D)
            for i in range(M):
                eps = torch.randn(B * depth, 1, D).cuda()
                eps_ = torch.bmm(eps, covar_half)
                lat = eps_.view(B, depth, D) + mu
                if i == 0:
                    latent = lat.unsqueeze(0)
                else:
                    latent = torch.cat((latent, lat.unsqueeze(0)), 0)
            return latent.view(B * M, depth, D)

    def get_distribution(self, x):
        x0_x = self.conv0_x(x)
        x1_x = self.conv1_x(self.pool(x0_x))
        x2_x = self.conv2_x(self.pool(x1_x))
        x3_x = self.conv3_x(self.pool(x2_x))
        x4_x = self.conv4_x(self.pool(x3_x))
        x4_ = self.conv_down(x4_x)

        x4_d = torch.transpose(x4_, 1, 2).flatten(start_dim=2)
        mean = self.mean_vec(x4_d)
        B, depth, D = mean.size()
        identity = torch.eye(D).unsqueeze(0).expand(B * depth, -1, -1).cuda() * 10.0
        covar_half_vec = self.covar_half(x4_d)
        covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)
        covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
        covar = torch.bmm(covar_half, covar_half)

        return mean, covar

    def kl_divergence(self, mu1, cov1, mu2, cov2):
        dist1 = MultivariateNormal(mu1, cov1)
        dist2 = MultivariateNormal(mu2, cov2)
        kl_div = torch.distributions.kl.kl_divergence(dist1, dist2)
        return kl_div

    def forward(self, input, fixed_imgs, fixed_masks, stage, M=1, num_reference=1):
        
        x0_x = self.conv0_x(input)
        x1_x = self.conv1_x(self.pool(x0_x))
        x2_x = self.conv2_x(self.pool(x1_x))
        x3_x = self.conv3_x(self.pool(x2_x))
        x4_x = self.conv4_x(self.pool(x3_x))
        x4_ = self.conv_down(x4_x)

        x4_d = torch.transpose(x4_, 1, 2).flatten(start_dim=2)
        mean = self.mean_vec(x4_d)
        B, depth, D = mean.size()
        identity = torch.eye(D).unsqueeze(0).expand(B * depth, -1, -1).cuda() * 10.0
        covar_half_vec = self.covar_half(x4_d)
        covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)
        covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + identity
        covar = torch.bmm(covar_half, covar_half) # 32, 256, 256
        # print(covar.shape)
        Z = self.reparameterize(mean, covar_half, M)
        Z_ = self.transform(Z).view(B * M, x4_.size()[-4], x4_.size()[-3], x4_.size()[-2], x4_.size()[-1])
        x4_x_size = [B * M, x4_x.size()[-4], x4_x.size()[-3], x4_x.size()[-2], x4_x.size()[-1]]

        x4_0_v = self.conv_up(Z_, output_size=x4_x.shape)

        x3_1x = self.conv3_1_x(torch.cat([x3_x, self.up(x4_0_v)], 1))
        x2_2x = self.conv2_2_x(torch.cat([x2_x, self.up(x3_1x)], 1))
        x1_3x = self.conv1_3_x(torch.cat([x1_x, self.up(x2_2x)], 1))
        x0_4x = self.conv0_4_x(torch.cat([x0_x, self.up(x1_3x)], 1))

        x_ori = self.tanh(self.conv00(x0_4x))

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_ = self.conv_down_(x4_0)

        Z_f = self.x_ori_vec(torch.transpose(x4_0_, 1, 2).flatten(start_dim=2))
        Z_f_ = self.transform_(Z_f).view(B, x4_0_.size()[-4], x4_0_.size()[-3], x4_0_.size()[-2],
                                         x4_0_.size()[-1]).repeat(M, 1, 1, 1, 1)

        x4_0_size = [B * M, x4_0.size()[-4], x4_0.size()[-3], x4_0.size()[-2], x4_0.size()[-1]]
        x4_0_z = self.conv_up_(Z_f_, output_size=x4_0_size)
        x4_00 = torch.cat([x4_0_z, x4_0_v], 1)

        outputs = None

        if stage == 1:
            id = random.randint(0, num_reference-1)
            fixed_img = fixed_imgs[:, id, :, :, :, :]
            fixed_mask = fixed_masks[:, id, 1, :, :, :].unsqueeze(1)

            f0_0 = self.conv0_0_(torch.cat([fixed_img, fixed_mask], dim=1))
            f1_0 = self.conv1_0(self.pool(f0_0))
            f2_0 = self.conv2_0(self.pool(f1_0))
            f3_0 = self.conv3_0(self.pool(f2_0))

            f2_0_ = f2_0
            f3_0_ = f3_0


            f_32_ = torch.mean(f2_0_, dim=2, keepdim=True).repeat(1, 1, depth, 1, 1)
            f_16_ = torch.mean(f3_0_, dim=2, keepdim=True).repeat(1, 1, depth, 1, 1)

            trans_32 = self.atten_32_(x2_0, f_32_)
            trans_16 = self.atten_16_(x3_0, f_16_)

            trans_32 = self.relu_32(self.in_32(trans_32))
            trans_16 = self.relu_16(self.in_16(trans_16))

            x3_1 = self.conv3_1(torch.cat([trans_16, self.up(x4_00)], 1))
            x2_2 = self.conv2_2(torch.cat([trans_32, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

            outputs = self.final(x0_4)

        else:

            output_list = []
            kl_list = []

            # v_shape = x4_0_v.shape  # B*M, C, D, W, H
            # x4_0_v = x4_0_v.view(M, B, v_shape[-4], v_shape[-3], v_shape[-2], v_shape[-1])

            for i in range(num_reference):
                
                fixed_img = fixed_imgs[:, i, :, :, :, :]
                fixed_mask = fixed_masks[:, i, 1, :, :, :].unsqueeze(1)

                f0_0 = self.conv0_0_(torch.cat([fixed_img, fixed_mask], dim=1))
                f1_0 = self.conv1_0(self.pool(f0_0))
                f2_0 = self.conv2_0(self.pool(f1_0))
                f3_0 = self.conv3_0(self.pool(f2_0))
                f4_0 = self.conv4_0(self.pool(f3_0))
                f4_0_ = self.conv_down_(f4_0) # (B, C, D, W, H)

                f2_0_ = f2_0
                f3_0_ = f3_0

                # f_32_avg = self.avg_pool_32(f2_0).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                # f_16_avg = self.avg_pool_16(f3_0).squeeze(-1).squeeze(-1).permute(0, 2, 1)

                # f_32_ = self.deep_fusion_32(f_32_avg, f2_0_)
                # f_16_ = self.deep_fusion_16(f_16_avg, f3_0_)

                f_32_ = torch.mean(f2_0_, dim=2, keepdim=True).repeat(1, 1, depth, 1, 1)
                f_16_ = torch.mean(f3_0_, dim=2, keepdim=True).repeat(1, 1, depth, 1, 1)

                trans_32 = self.atten_32_(x2_0, f_32_)
                trans_16 = self.atten_16_(x3_0, f_16_)

                trans_32 = self.relu_32(self.in_32(trans_32))
                trans_16 = self.relu_16(self.in_16(trans_16))

                x3_1 = self.conv3_1(torch.cat([trans_16, self.up(x4_00)], 1))
                x2_2 = self.conv2_2(torch.cat([trans_32, self.up(x3_1)], 1))
                x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
                x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

                output = self.final(x0_4)
                output_pseudo = torch.softmax(output, dim=1).detach()

                with torch.no_grad():
                    mean_fixed, covar_fixed = self.get_distribution(fixed_img)

                sum_kl = 0.
                mean_v = torch.mean(mean, dim=1)
                covar_v = torch.mean(covar, dim=0)
                mean_ref = torch.mean(mean_fixed, dim=1)
                covar_ref = torch.mean(covar_fixed, dim=0)
                sum_kl = self.kl_divergence(mean_v, covar_v, mean_ref, covar_ref)
                
                # for d in range(depth):
                #     kl = self.kl_divergence(mean[:, d, :], covar[d, :, :], mean_fixed[:, d, :], covar_fixed[d, :, :])
                #     sum_kl += kl

                # sum_kl = sum_kl/depth

                # mean_f4_0_ = torch.mean(f4_0_, dim=2)
                # maen_x4_0_ = torch.mean(x4_0_, dim=2)
                # similarity = self.cosine_similarity(mean_f4_0_, maen_x4_0_)

                output_list.append(output_pseudo)
                kl_list.append(sum_kl)

                # if outputs == None:
                #     outputs = output_pseudo
                # else:
                #     outputs = outputs + output_pseudo
            stacked = torch.stack(kl_list)
            meankld = torch.mean(stacked, dim=0)
            ids = len(output_list)
            sum_w = 0.0

            for id in range(ids):

                output_pseudo = output_list[id]
                kl = kl_list[id] - meankld
                un_certainty = self.uncertainty_map(output_pseudo)
                # weight = torch.exp(-un_certainty) / (1 + torch.exp(-sim))
                sigmoid = nn.Sigmoid()
                weight = torch.exp(-un_certainty)*(sigmoid(-alpha*kl)+0.5)
                sum_w += weight
                
                if outputs == None:
                    outputs = output_pseudo * weight
                else:
                    outputs = outputs + output_pseudo * weight

            outputs = outputs / sum_w

        return x_ori, outputs, mean, covar.view(-1, depth, D, D),


