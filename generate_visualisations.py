# Generate visualizations of saliency maps for ecgs
# Parts of the implementation bellow is adapted from that of:
# http://kazuto1011.github.io from Kazuto Nakashima.
# Which is made available under MIT license.

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import ECGDatasetH5

import os
import numpy as np

from resnet import ResNet1d
import h5py
import json


def intensity_from_grad(grad, scale=1):
    return scale * grad ** 2 / max((grad ** 2).mean(), 0.001)


class BackPropagation(object):
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def __call__(self, x):
        x.requires_grad = True
        self.x = x
        self.x_shape = x.shape[2:]
        self.logits = self.model(x)
        return self.logits

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def generate(self):
        visualization_loss = self.logits[0]
        x_grad, = torch.autograd.grad(visualization_loss, self.x)
        return x_grad[0]


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """
    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(BackPropagation):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                if not module._modules:  # Ignore layers that are  containers of sublayers
                    self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                    self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        _x_grad = super(GradCAM, self).generate()
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.x_shape, mode="linear", align_corners=False
        )

        B, C, L = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        if gcam.max() > 0:
            gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, L)

        return torch.stack([gcam[0, 0]]*N_LEADS)  # Repeat the same information for all layers



if __name__ == "__main__":
    import argparse
    import ecg_plot
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt



    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_traces', default='../data/samitrop/samitrop1631.hdf5',
                        help='path to data.')
    parser.add_argument('--path_to_chagas', default='../data/chagas_samitrop.csv',
                        help='path to data.')
    parser.add_argument('--model', default='./model/model12',
                        help='path to data.')
    parser.add_argument('--traces_dset', default='tracings',
                     help='traces dataset in the hdf5 file.')
    parser.add_argument('--examid_dset', default='exam_id',
                     help='exam id dataset in the hdf5 file.')
    parser.add_argument('-i', '--ith_element_in_class', type=int, default=0,
                        help='pick the i-th element of the given class to analize.'
                             'Exam id has priority ovr this option. I.e., if a specifc exam is passed through '
                             ' the command line this option is ignored.')
    parser.add_argument('-c', '--example_class', choices=[0, 1, 2], type=int, default=0,
                        help='pick the i-th element of the given class to analize')
    parser.add_argument('--save', type=str, default=os.path.join(parser.parse_args().model, 'heat_maps'),
                        help='file to save the plot. Otherwise just save on the ')
    args = parser.parse_args()

    # Load model
    with open(os.path.join(args.model, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    ckpt = torch.load(os.path.join(args.model, 'model.pth'), map_location=lambda storage, loc: storage)

    # Get model
    N_LEADS = 12
    N_CLASSES = 1  # two classes, but just need one output
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=N_CLASSES,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    model.load_state_dict(ckpt["model"])
    model = model.eval()

    # Load data
    test_set = ECGDatasetH5(
        path=args.path_to_traces,
        traces_dset=args.traces_dset,
        exam_id_dset=args.examid_dset,
        ids_dset=None,
        path_to_chagas=args.path_to_chagas
        )

    eval_file = os.path.join(args.model, 'evaluation.csv')
    df_eval = pd.read_csv(eval_file)
    data_true = df_eval['test_true'].to_numpy()
    data_output = df_eval['test_output'].to_numpy()
    sort_index = np.argsort(data_output)
    chagas_diag = test_set.chagas
    if len(test_set)!=df_eval.shape[0]:
        raise Exception("Mismatch! Evaluation file from different test set?")

    n_samples = 50
    ind1 = sort_index[np.where(chagas_diag[sort_index]==0)[0]][:n_samples]  # lowest non-chagas
    ind2 = sort_index[np.where(chagas_diag[sort_index]==1)[0]][:n_samples]  # lowest chagas
    ind3 = sort_index[np.where(chagas_diag[sort_index]==0)[0]][-n_samples:]  # highest non-chagas
    ind4 = sort_index[np.where(chagas_diag[sort_index]==1)[0]][-n_samples:]  # highest chagas
    interesting_indices = np.concatenate((ind1, ind2, ind3, ind4))

    res_blocks = ['resblock1d_2.conv1']#, 'resblock1d_2.conv1', 'resblock1d_3.conv1', 'resblock1d_4.conv1']
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # TODO: save id
    for res_block in res_blocks:
        for w, idx in enumerate(interesting_indices):
            if w%(4*n_samples)<n_samples:
                name = 'lowestNonChagas'
                continue
            elif w%(4*n_samples)<2*n_samples:
                name = 'lowestChagas'
                continue
            elif w%(4*n_samples)<3*n_samples:
                name = 'NonChagas'#'highestNonChagas'
            else:
                name = 'Chagas'#'highestChagas'

            xi, _ = test_set.getbatch(idx, idx+1)

            xi.requires_grad = True

            yi_predicted = model(xi)

            probs = torch.sigmoid(yi_predicted).cpu().detach().numpy().flatten()
            print('P(chagas) = {:0.3f}'.format(*probs))  # grain of salt!! assumes balanced data set ...

            grad_cam_model = GradCAM(model)
            # guided_back_model_model = GuidedBackPropagation(model)

            _ = grad_cam_model(xi)  # not using the result, but need it for initialisation

            xi_grad = grad_cam_model.generate(res_block)
            # xi_grad = guided_back_model.generate()

            lead = ['I', 'II', 'III', 'AVL', 'AVF', 'AVR', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

            # Plot using ecgplot
            s = xi[0].detach().cpu().data.numpy()
            g = xi_grad.detach().cpu().data.numpy()
            row_height = 24
            cols = 2
            ecg_plot.plot(s, sample_rate=400, style='bw', row_height=row_height, lead_index=lead, columns=cols)
            rows = len(lead) // cols
            sample_rate = 400
            scale = 4.0
            for i in range(rows * cols):
                row = i % rows
                column = i // rows
                y_offset = -(row_height / 2) * row
                x_offset = (len(s[0]) / sample_rate) * column
                gp = intensity_from_grad(g[i, :], scale)
                plt.scatter(x_offset + np.arange(len(s[i, :])) / sample_rate, y_offset + s[i, :], marker='o',
                            color='plum', s=gp)
            if args.save is not None:
                plt.savefig(os.path.join(args.save, 'heat_map_'+res_block+'_'+name+'_p'+str(*probs.round(3))+'_id'+str(df_eval[args.examid_dset][idx])+'.pdf'))
            else:
                plt.show()
