import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
from easydict import EasyDict

from .builder import SPPE
from .layers.real_nvp import RealNVP
from .layers.Resnet import ResNet

NUM_OF_JOINTS = 17

# neural network for RealNVP
def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())
def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))

def nets_m():
    return nn.Sequential(nn.Linear(2 * NUM_OF_JOINTS, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2 * NUM_OF_JOINTS), nn.Tanh())
def nett_m():
    return nn.Sequential(nn.Linear(2 * NUM_OF_JOINTS, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2 * NUM_OF_JOINTS))


class Linear(nn.Module):
    '''Linear layer with optional weight normalization and bias'''
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01) # Xavier weight initialization

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


@SPPE.register_module
class RegressFlow(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(RegressFlow, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048
        }[cfg['NUM_LAYERS']]
        self.hidden_list = cfg['HIDDEN_LIST']

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs, out_channel = self._make_fc_layer()

        self.fc_coord = Linear(out_channel, self.num_joints * 2)
        self.fc_sigma = Linear(out_channel, self.num_joints * 2, norm=False)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        # prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        # masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        # self.flow = RealNVP(nets, nett, masks, prior)
        
        prior_m = distributions.MultivariateNormal(torch.zeros(2 * NUM_OF_JOINTS), torch.eye(2 * NUM_OF_JOINTS))
        np_mask_m = np.zeros((2, 2 * NUM_OF_JOINTS))
        for i in range(2):
            for j in range(i%2, 2 * NUM_OF_JOINTS, 2):
                np_mask_m[i, j] = 1
        masks_m = torch.from_numpy(np.array(np_mask_m.tolist() * 3).astype(np.float32))
        
        self.flow_m = RealNVP(nets_m, nett_m, masks_m, prior_m)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())

        return nn.Sequential(*fc_layers), input_channel

    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 2)
        assert out_coord.shape[2] == 2

        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 2)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid()
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        if self.training and labels is not None:
            gt_uv = labels['target_uv'].reshape(pred_jts.shape)
            bar_mu = (pred_jts - gt_uv) / sigma
            bar_mu_adjusted = bar_mu * labels['target_uv_weight'].reshape(BATCH_SIZE, self.num_joints, 2)
            
            # compute the log probability of the data, marginalizing out the unseen joints
            log_phi = self.flow_m.log_prob(bar_mu_adjusted.reshape(-1, 2 * NUM_OF_JOINTS),
                                           mask = labels['target_uv_weight'].reshape(BATCH_SIZE, self.num_joints, 2)
                                           ).reshape(BATCH_SIZE, 1, 1)
            # num_of_seen_data = torch.sum(labels['target_uv_weight'].reshape(BATCH_SIZE, -1), dim=1)
            # print(f'\nlog_phi: {log_phi}, weight: ', num_of_seen_data)
            
            # distribute the probability to each point
            num_of_seen_data = torch.sum(
                labels['target_uv_weight'].reshape(BATCH_SIZE, -1), dim=1
                ).reshape(BATCH_SIZE, 1, 1)
            nf_loss = torch.log(sigma) - log_phi / num_of_seen_data
        else:
            nf_loss = None

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )
        return output
