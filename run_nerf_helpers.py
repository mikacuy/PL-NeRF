'''
Mikaela Uy
mikacuy@cs.stanford.edu

PL-NeRF: novel view synthesis experiments
A piecewise linear formulation to volume rendering
'''

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16 - 1) * np.clip(x,0,1)).astype(np.uint16)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def compute_space_carving_loss_corrected(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    n_rays, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    if target_hypothesis.shape[-1] == 1:
        ### In the case where there is no caching of quantiles
        target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)
    else:
        ### Each quantile here already picked a hypothesis
        target_hypothesis_repeated = target_hypothesis

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        quantile_mean = torch.mean(distances, axis=1) ## mean for each quantile, averaged across all rays
        samples_min = torch.min(quantile_mean, axis=0)[0]
        loss =  torch.mean(samples_min, axis=-1)


    else:
        ### Each ray selects a hypothesis
        best_hyp = torch.min(distances, dim=0)[0]   ## for each sample pick a hypothesis
        ray_mean = torch.mean(best_hyp, dim=-1) ## average across samples
        loss = torch.mean(ray_mean)  

    return loss

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sample_pdf_return_u(bins, weights, N_samples, det=False, pytest=False, load_u=None):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    if load_u is None:
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

    ## Otherwise, take the saved u
    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples, u


def pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )

    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_right - tau_left))
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))

    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)

    sample = s_left + t

    return sample


def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, quad_solution_v2=False, zero_threshold = 1e-4, epsilon_=1e-3):

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]

    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)

    samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)

    ### Also return these for custom autograd
    ### T_below, tau_below, bin_below
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below


def sample_pdf_reformulation_return_u(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, load_u=None, quad_solution_v2=True, zero_threshold = 1e-4, epsilon_=1e-3):
    

    bins = torch.cat([near, bins, far], -1)   
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    if load_u is None:
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)
    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]
    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    # zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    ### Increasing
    samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)

    ### Decreasing
    samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)

    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below, u


#### Utils for eval #####
def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())


class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (self.mean_dict[key] * self.total_weight + l) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return (key in self.mean_dict)

    def get(self, key):
        return self.mean_dict[key]
    
    def as_dict(self):
        return self.mean_dict
        
    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0
    
    def print(self, f=None):
        for key, l in self.mean_dict.items():
            if f is not None:
                print("{}: {}".format(key, l), file=f)
            else:
                print("{}: {}".format(key, l))

    
