"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from cv2 import CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

# REMEMBER!!! Steps go from 100 to 0 (100 first step, 0 last step)
SCHEDULERS = {
    "constant": lambda timesteps, num_timesteps: th.ones(timesteps.shape[0], device=timesteps.device),
    "linear": lambda timesteps, num_timesteps: 1 - timesteps / num_timesteps,
    "linear_inverse": lambda timesteps, num_timesteps: timesteps / num_timesteps,
    "sqrt": lambda timesteps, num_timesteps: 1 - th.sqrt(timesteps / num_timesteps),
    "log": lambda timesteps, num_timesteps: 1 - th.log(timesteps + 1) / np.log(num_timesteps + 1),
    "cosine": lambda timesteps, num_timesteps: th.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2,
    "cosine_inverse": lambda timesteps, num_timesteps: 1-th.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2,
    "step": lambda timesteps, num_timesteps: (timesteps <= 10).type(th.float) - (timesteps >=90).type(th.float),
    "sqrt1e-4_inverse": lambda timesteps, num_timesteps: 1-th.sqrt((1- timesteps / num_timesteps) + 0.0001),
}


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "linear_v2":
        linear_start, linear_end = 1e-4, 2e-2
        return (th.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype=th.float64) ** 2)
    
    elif schedule_name == "linear_v3":
        linear_start, linear_end = 0.0015, 0.015
        return (th.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype=th.float64) ** 2)
    
    elif schedule_name == "linear_v4":
        linear_start, linear_end = 1e-4, 2e-3
        return (th.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype=th.float64) ** 2)
    
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif 'sqrt' in schedule_name:
        sqrt_schedulers = {
            "10sqrt1e-4": lambda t: max(0, 1-np.power(t + 0.0001, 1/10)),
            "5sqrt1e-4": lambda t: max(0, 1-np.power(t + 0.0001, 1/5)),
            "3sqrt1e-4": lambda t: max(0, 1-np.power(t + 0.0001, 1/3)),
            "sqrt1e-4": lambda t: max(0, 1-np.sqrt(t + 0.0001)),
            "sqrt2e-2": lambda t: max(0, 1-np.sqrt(t + 0.02)),
            "sqrt5e-2": lambda t: max(0, 1-np.sqrt(t + 0.05)),
            "sqrt1e-1": lambda t: max(0, 1-np.sqrt(t + 0.1)),
            "sqrt2e-2": lambda t: max(0, 1-np.sqrt(t + 0.2)),    
        }
        assert schedule_name in sqrt_schedulers.keys(), f"Unknown sqrt scheduler {schedule_name}"
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            sqrt_schedulers[schedule_name],
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / max(0.00001, alpha_bar(t1)), max_beta)) # the max is to prevent singularities
    return np.array(betas)

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

mean_type_dict = {
    "previous_x": ModelMeanType.PREVIOUS_X,
    "start_x": ModelMeanType.START_X,
    "epsilon": ModelMeanType.EPSILON
}


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

var_type_dict = {
    "learned": ModelVarType.LEARNED,
    "fixed_small": ModelVarType.FIXED_SMALL,
    "fixed_large": ModelVarType.FIXED_LARGE,
    "learned_range": ModelVarType.LEARNED_RANGE
}

LOSSES_TYPES = ["mse", "rescaled_mse", "kl", "rescaled_kl", "mse_v", "apd", "motion", "mm_mse", "mm_mse_l1", "weighed_mse", "log_dif_mse", "mse_l1", "lossy"]
MSE, RESCALED_MSE, KL, RESCALED_KL, MSE_V, APD, MOTION, MM_MSE, MM_MSE_L1, WEIGHED_MSE, LOG_DIF_MSE, MSE_L1, LOSSY = LOSSES_TYPES

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        noise_schedule,
        steps,
        predict='start_x',
        var_type="fixed_large",
        losses="mse", # this can be a string or an array of strings
        losses_multipliers=1., # this can be a single float or an array of floats
        rescale_timesteps=False,
        noise_std=1,
        **kwargs
    ):
        assert predict in mean_type_dict.keys(), f"predict='{predict}' not supported"
        self.model_mean_type = mean_type_dict[predict]
        assert var_type in var_type_dict.keys(), f"var_type='{var_type}' not supported"
        self.model_var_type = var_type_dict[var_type]

        # support for a linear combination (losses_multipliers)) of several loses
        if isinstance(losses, str):
            losses = [losses, ] # retro-compatibility
        if isinstance(losses_multipliers, float):
            losses_multipliers = [losses_multipliers, ]
        assert len(losses) == len(losses_multipliers)
        for loss_type in losses:
            assert loss_type in LOSSES_TYPES, f"loss_type='{loss_type}' not supported"
        self.losses = losses
        self.losses_multipliers = losses_multipliers

        self.rescale_timesteps = rescale_timesteps
        self.noise_std = noise_std

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(noise_schedule, steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def process_model_output_before_loss(self, model, out, model_kwargs):
        return out # needs to be overriden if sth else needs to be done

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0). 
        -> using the reparametrization trick of:
            sqrt(alfa) * x_0 + sqrt(1 - alfa) * eps

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std
        assert noise.shape == x_start.shape
        weighed_x_start = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        weighed_noise = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return ( weighed_x_start + weighed_noise )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x) * self.noise_std
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        noise_to_add = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        sample = out["mean"] + noise_to_add
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "noise_to_add": noise_to_add}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        max_step=-1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param max_step: last diffusion step that wants to perform. If -1, all steps are performed
        :return: a non-differentiable batch of samples.
        """
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )):
            if max_step != -1 and i == max_step:
                break
            final = sample

        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x) * self.noise_std
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, all_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std
        x_t = self.q_sample(x_start, t, noise=noise) # apply perturbations from '0' to 't' to original image (x_start)

        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
        model_output = self.process_model_output_before_loss(model, model_output, all_kwargs)
        target = self.process_model_output_before_loss(model, target, all_kwargs)
        assert model_output.shape == target.shape# == x_start.shape

        # kl divergence loss
        if KL in self.losses or RESCALED_KL in self.losses:
            terms[KL] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if RESCALED_KL in self.losses:
                terms[RESCALED_KL] *= self.num_timesteps

        if MSE in self.losses or RESCALED_MSE in self.losses:
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if RESCALED_MSE in self.losses:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            mse = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                if MSE in self.losses:
                    terms[MSE] = mse + terms["vb"]
                if RESCALED_MSE in self.losses:
                    terms[RESCALED_MSE] = mse + terms["vb"]
            else:
                if MSE in self.losses:
                    terms[MSE] = mse
                if RESCALED_MSE in self.losses:
                    terms[RESCALED_MSE] = mse # they are the same when variance is not learned

        if MSE_V in self.losses:
            assert "obs" in model_kwargs, f"Observation window must be provided for '{MSE_V}' loss computation."
            last_obs = model_kwargs["obs"][:,-1]
            first_pred = model_output[:,0]
            terms[MSE_V] = mean_flat((first_pred - last_obs) ** 2)

        # Motion
        if MOTION in self.losses:
            terms[MOTION] = 1 / model_output.mean()
       
        # multipliers => loss
        terms["loss"] = 0
        for loss_key, multiplier in zip(self.losses, self.losses_multipliers):
            terms["loss"] += terms[loss_key] * multiplier

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start) * self.noise_std
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def get_gt(
        self,
        model,
        obs,
        pred
    ): 
        raise NotImplementedError


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



class LatentDiffusion(GaussianDiffusion):

    def __init__(
        self,
        *,
        noise_schedule,
        steps,
        predict='start_x',
        var_type="fixed_large",
        losses=["mse", ], # this can be a string or an array of strings
        losses_multipliers=[1.,], # this can be a single float or an array of floats
        losses_schedulers=["constant", ],
        losses_decoded=[False, ], # this can be an array of booleans
        alpha=100,
        p=1, # pdist --> degree
        k=10,
        mmgt_samples=12,
        rescale_timesteps=False,
        noise_std=1,
        cond_dropout=0.0,
        use_mm_gt=False,
        **kwargs
    ):
        super().__init__(noise_schedule=noise_schedule,
            steps=steps,
            predict=predict,
            var_type=var_type,
            losses=losses, # this can be a string or an array of strings
            losses_multipliers=losses_multipliers, # this can be a single float or an array of floats
            rescale_timesteps=rescale_timesteps,
            noise_std=noise_std,
            **kwargs
        )
        self.alpha = alpha
        self.k = k
        self.p = p
        self.mmgt_samples = mmgt_samples
        self.use_mm_gt = use_mm_gt
        assert self.use_mm_gt in [False, "sample", "all"], "use_mm_gt must be one of [False, 'sample', 'all']"
        self.cond_dropout = cond_dropout
        for sch in losses_schedulers:
            assert sch in SCHEDULERS, "all schedulers must be in {}".format(list(SCHEDULERS.keys()))
            
        self.losses_schedulers = [SCHEDULERS[sch] for sch in losses_schedulers]
        self.losses_decoded = losses_decoded

    def process_model_output_before_loss(self, model, out, all_kwargs):
        if self.decoded_loss:
            assert "obs" in all_kwargs, "obs must be in all_kwargs"
            return model.decode_pred(all_kwargs["obs"], out)
        return super().process_model_output_before_loss(model, out, all_kwargs)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred=None,
    ): 

        # ---------
        shape = (shape[0], model.get_emb_size())
        obs_orig = model_kwargs["obs"]
        # copy of model_kwargs
        model_kwargs = model_kwargs.copy()
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        #pred_emb = model.encode_pred(pred) # USEFUL TO TEST
        # ---------

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1] # WARNING: from step 99 to step 0!!!

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        #previous_out = None
        for i in indices:
            #if i < 50:
            #    for key in out:
            #        out[key] = th.zeros_like(out[key])
            #    yield out
             #   continue
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            
                out["pred_xstart_enc"] = out["pred_xstart"]
                out["sample_enc"] = out["sample"]
                out["sample"] = model.decode_pred(obs_orig, out["sample"])
                out["pred_xstart"] = model.decode_pred(obs_orig, out["pred_xstart"])

                #if improved_sampling and previous_out is not None: # improved sampling from Cold Diffusion starting from second step
                    #img = img - previous_out["sample_enc"] + out["sample_enc"] # wrong. D(x0, s) must use current x0 approximation
                """
                # yields nans... it keeps increasing the magnitude of img until it blows up
                updated_diffused_prev = out["pred_xstart_enc"] + previous_out["noise_to_add"]
                img = img - updated_diffused_prev + out["sample_enc"]
                """
                #    img = img - self.q_sample(out["pred_xstart_enc"], t) + self.q_sample(out["pred_xstart_enc"], t-1)
                #else:
                img = out["sample_enc"]

                #previous_out = out
                yield out

    def p_sample_loop_progressive_always_GT(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred=None
    ): 

        # ---------
        shape = (shape[0], model.get_emb_size())
        obs_orig = model_kwargs["obs"]
        # copy of model_kwargs
        model_kwargs = model_kwargs.copy()
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        pred_emb = model.encode_pred(pred, obs_orig) # USEFUL TO TEST --> it is the GT
        # ---------

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1] # WARNING: from step 99 to step 0!!!

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            #if i < 50:
            #    for key in out:
            #        out[key] = th.zeros_like(out[key])
            #    yield out
             #   continue
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            
                img = self.q_sample(pred_emb, t)#out["sample"]

                out["pred_xstart_enc"] = out["pred_xstart"]
                out["sample"] = model.decode_pred(obs_orig, out["sample"])
                out["pred_xstart"] = model.decode_pred(obs_orig, out["pred_xstart"])
                yield out

    def p_sample_loop_progressive_test_condition(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred=None
    ): 

        # ---------
        shape = (shape[0], model.get_emb_size())
        obs_orig = model_kwargs["obs"]
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        model_kwargs_zero = {
            "obs": th.zeros_like(model_kwargs["obs"])
        }
        # ---------

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1] # WARNING: from step 99 to step 0!!!

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs)
                out_nocondition = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs_zero)
            
                img = out["sample"] # input of next timestep is the conditioned one

                out["pred_xstart_enc"] = out_nocondition["pred_xstart"]
                out["sample"] = model.decode_pred(obs_orig, out_nocondition["sample"])
                out["pred_xstart"] = model.decode_pred(obs_orig, out_nocondition["pred_xstart"])
                yield out

    def p_sample_loop_progressive_test_noise(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred=None
    ): 

        # ---------
        shape = (shape[0], model.get_emb_size())
        obs_orig = model_kwargs["obs"]
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        # ---------

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            noise_original = noise
        else:
            noise_original = th.randn(*shape, device=device) * self.noise_std
        img = noise_original
        indices = list(range(self.num_timesteps))[::-1] # WARNING: from step 99 to step 0!!!

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs)
                out_noise = self.p_sample(model, noise_original, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs)
            
                img = out["sample"] # input of next timestep is the conditioned one

                out["pred_xstart_enc"] = out_noise["pred_xstart"]
                out["sample"] = model.decode_pred(obs_orig, out_noise["sample"])
                out["pred_xstart"] = model.decode_pred(obs_orig, out_noise["pred_xstart"])
                yield out

   
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        pred=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        # ---------
        shape = (shape[0], model.get_emb_size())
        obs_orig = model_kwargs["obs"]
        # copy of model_kwargs
        model_kwargs = model_kwargs.copy()
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        #pred_emb = model.encode_pred(pred) # USEFUL TO TEST
        # ---------


        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device) * self.noise_std
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
            
                out["pred_xstart_enc"] = out["pred_xstart"]
                out["sample_enc"] = out["sample"]
                out["sample"] = model.decode_pred(obs_orig, out["sample"])
                out["pred_xstart"] = model.decode_pred(obs_orig, out["pred_xstart"])
                
                yield out
                img = out["sample_enc"]

    def get_gt(
        self,
        model,
        obs,
        pred
    ): 

        pred_emb = model.encode_pred(pred, obs) # USEFUL TO TEST
        return model.decode_pred(obs, pred_emb)


    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, all_kwargs=None):
        batch_size =  x_start.shape[0]
        original_obs = model_kwargs["obs"]
        original_obs_expanded = model_kwargs["obs"].repeat_interleave(self.k, dim=0)
        if self.use_mm_gt == "all":
            # target x_start is the multimodal ground truth
            assert "mm_gt" in all_kwargs, "use_mm_gt is True but mm_gt is not in all_kwargs"
            # retrieve K multimodal ground truths
            mm_gt = all_kwargs["mm_gt"]
            assert mm_gt.shape[1] >= self.k, "not enough multimodal ground truth samples for k={}".format(self.k)
            mm_gt = mm_gt[:, :self.k]
            x_start = mm_gt.reshape([-1, ] + list(mm_gt.shape[2:]))
            x_start = model.encode_pred(x_start, original_obs_expanded)
        elif self.use_mm_gt == "sample":
            # we sample a single GT from the multimodal GT, and use it as the x_start
            assert "mm_gt" in all_kwargs, "use_mm_gt is True but mm_gt is not in all_kwargs"
            # randomly select a single multimodal ground truth
            mm_gt = all_kwargs["mm_gt"]
            # sample a single multimodal ground truth for each sample in the batch
            idces = th.randint(0, mm_gt.shape[1], (batch_size, ))
            mm_gt = mm_gt[th.arange(batch_size), idces]
            x_start = model.encode_pred(mm_gt, original_obs)
            x_start = x_start.repeat_interleave(self.k, dim=0) # repeat K times
        else:
            # target x_start is the unimodal and REAL ground truth
            original_enc_x_start = model.encode_pred(x_start, original_obs)
            x_start = original_enc_x_start.repeat_interleave(self.k, dim=0) # repeat K times

        # replace obs with zeros with probability "self.cond_dropout"
        model_kwargs["obs"] = model.encode_obs(model_kwargs["obs"])
        if np.random.random() < self.cond_dropout: # for classifier-free guidance!
            model_kwargs["obs"] = th.zeros_like(model_kwargs["obs"])

        #print(model_kwargs["obs"].shape, all_kwargs["obs"].shape, x_start.shape, t.shape)
        # we repeat the obs 'k' times to generate 'k' predictions for each observation
        model_kwargs["obs"] = model_kwargs["obs"].repeat_interleave(self.k, dim=0)
        all_kwargs["obs"] = all_kwargs["obs"].repeat_interleave(self.k, dim=0)

        target_x_start = x_start
        orig_t = t
        t = t.repeat_interleave(self.k, dim=0) # (2, 25) -> (2, ..., 2, 25, ..., 25)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) * self.noise_std

        x_t = self.q_sample(x_start, t, noise=noise) # apply perturbations from '0' to 't' to original image (x_start)

        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=target_x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: target_x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

        # check if any losses_decoded is True
        if any(self.losses_decoded) or WEIGHED_MSE in self.losses or LOSSY in self.losses:
            decoded_model_output = model.decode_pred(all_kwargs["obs"], model_output).contiguous()
            decoded_target = model.decode_pred(all_kwargs["obs"], target).contiguous().detach()

            assert decoded_model_output.shape == decoded_target.shape# == x_start.shape


        terms["loss"] = 0
        assert len(self.losses) == len(self.losses_multipliers) == len(self.losses_schedulers) == len(self.losses_decoded)
        for loss_key, multiplier, scheduler, decoded in zip(self.losses, self.losses_multipliers, self.losses_schedulers, self.losses_decoded):
            if decoded:
                current_pred = decoded_model_output
                current_target = decoded_target
                suffix = "_d"
            else:
                current_pred = model_output
                current_target = target
                suffix = ""

            if MSE == loss_key:
                # ADE definition
                diff = current_pred.reshape(batch_size, self.k, -1) - current_target.reshape(batch_size, self.k, -1)
                terms[MSE+suffix] = th.linalg.norm(diff, axis=-1).min(axis=-1).values

            elif MSE_L1 == loss_key:
                # ADE definition
                diff = model_output.reshape(batch_size, self.k, -1) - target.reshape(batch_size, self.k, -1)
                terms[MSE_L1+suffix] = th.absolute(diff).sum(axis=-1).min(axis=-1).values
                

            # multipliers => loss
            terms["loss"] += (terms[loss_key + suffix] * multiplier * scheduler(orig_t, self.num_timesteps))#.mean()

        # NOTE: losses are not averaged. This is done in the trainer.
        return terms






    
