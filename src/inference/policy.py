from typing import Any
from einops import rearrange, repeat
import torch
from core.specs import TimedActionTrajectoryNormalized, Observation, TimedActionTrajectory
from torch import Tensor
from jaxtyping import Float
from abc import ABC, abstractmethod

from flow_matching.solver.ode_solver import ODESolver
from flow_matching.solver.sde_solver import SDESolver
from flow_matching.utils.model_wrapper import ModelWrapper
from training.processor import ActionTrajectoryNormalizer, ObservationProcessor


class Policy:
    def __init__(
        self,
        ode_method: str,
        ode_opts: dict[str, float],
        actions_shape: tuple[int, ...],
        model_wrapper: ModelWrapper,
        action_normalizer: ActionTrajectoryNormalizer,
        observation_processor: ObservationProcessor,
    ):
        self.actions_shape = actions_shape
        self.model_wrapper = model_wrapper
        self.action_normalizer = action_normalizer
        self.observation_processor = observation_processor

        self.solver = ODESolver(velocity_model=self.model_wrapper)
        
        self.sample_params = {
            'method': ode_method,
            'atol': ode_opts['atol'] if 'atol' in ode_opts else 1e-5,
            'rtol': ode_opts['rtol'] if 'rtol' in ode_opts else 1e-5,
            'step_size': ode_opts['step_size'] if 'step_size' in ode_opts else None,
            'return_intermediates': False,
        }
        
        self.sde_solver = SDESolver(velocity_model=self.model_wrapper)


    def infer(
        self,
        observations: Observation,
        temperature = 0.0,
    ) -> tuple[TimedActionTrajectory, dict[str, Any]]:
        
        info = {}

        training = self.model_wrapper.training
        self.model_wrapper.train(False)

        device = observations.device

        conditioning = self.observation_processor.process(observations).observations

        x_0 = torch.randn(observations.batch_size + self.actions_shape, device=device)
        
        # TODO: get rid of eps
        eps = 1e-5
        synthetic_samples = self.solver.sample(
            x_init=x_0,
            time_grid=torch.tensor([0.0, 1.0], device=device),
            extra={"conditioning": conditioning},
            **self.sample_params,
        ).clamp(min=-1.0 + eps, max=1.0 - eps)
        
        if temperature > 0.0:
            batch_size = synthetic_samples.shape[0]
            synthetic_samples_similar = self._generate_similar(
                sample=rearrange(synthetic_samples, 'b c h -> (b c) 1 h'),
                similar_samples_n=3,
                temperature=temperature,
            ).clamp(min=-1.0 + eps, max=1.0 - eps)
            synthetic_samples_similar = rearrange(
                synthetic_samples_similar, '(b c) s 1 h -> b s c h', b=batch_size
            )
            
            info['synthetic_samples'] = synthetic_samples.cpu()
            info['synthetic_samples_similar'] = synthetic_samples_similar.cpu()

            synthetic_samples = synthetic_samples_similar[:, 0]
            
        synthetic_actions = self.action_normalizer.unnormalize(
            TimedActionTrajectoryNormalized.from_unified(synthetic_samples)
        )

        self.model_wrapper.train(training)

        return synthetic_actions, info
    
    def _generate_similar(
        self,
        sample: Float[Tensor, 'batch 1 horizon'],
        similar_samples_n: int,
        temperature: float = 0.0,
    ) -> Float[Tensor, 'batch samples 1 horizon']:

        sample = repeat(sample, 'b 1 h -> b s 1 h', s=similar_samples_n)
        
        batch_size, _, _, horizon_n = sample.shape
        dev = sample.device
        
        x = torch.arange(horizon_n, device=dev)
        
        centers = torch.randint(
            0,
            horizon_n,
            (batch_size, similar_samples_n, 1),
            device=dev,
        )
        widths = torch.randint(
            horizon_n // 16,
            horizon_n // 4,
            (batch_size, similar_samples_n, 1),
            device=dev,
        ).float()
        amplitudes = (
            torch.rand(batch_size, similar_samples_n, 1, 1, device=dev) * 2 - 1
        ) * temperature
        
        x_expanded = rearrange(x, 'h -> 1 1 1 h')
        centers_expanded = rearrange(centers, 'b s 1 -> b s 1 1')
        widths_expanded = rearrange(widths, 'b s 1 -> b s 1 1')
        
        gaussian = amplitudes * torch.exp(
            -((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2)
        )
        
        sample = sample + gaussian
        return sample.clamp(min=-1.0, max=1.0)        

    def stochastic_infer(
        self,
        observations: Observation,
        noise_level: float = 1.0,
    ) -> tuple[TimedActionTrajectory, dict[str, Any]]:
        
        device = observations.device

        training = self.model_wrapper.training
        self.model_wrapper.train(False)
        
        conditioning = self.observation_processor.process(observations).observations
        
        x_0 = torch.randn(observations.batch_size + self.actions_shape, device=device)
        
        eps = 1e-5
        samples, intermediates = self.sde_solver.sample(
            x_init=x_0,
            step_size=self.sample_params['step_size'],
            return_intermediates=True,
            extra={"conditioning": conditioning},
            noise_level=noise_level,
        )
        
        samples = samples.clamp(min=-1.0 + eps, max=1.0 - eps)
        
        samples = self.action_normalizer.unnormalize(
            TimedActionTrajectoryNormalized.from_unified(samples)
        )
        
        self.model_wrapper.train(training)
        
        return samples, intermediates
        
    def stochastic_posterior_gaussian(
        self,
        x_t_unified: Float[Tensor, 'b c h'],
        t: Float[Tensor, 'b'],
        observations: Observation,
        noise_level: float = 1.0,
    ) -> tuple[Float[Tensor, 'b c h'], Float[Tensor, 'b']]:

        training = self.model_wrapper.training
        self.model_wrapper.train(False)
        
        conditioning = self.observation_processor.process(observations).observations
        
        posterior_mean, posterior_var = self.sde_solver.posterior_gaussian(
            x_t=x_t_unified,
            t=t,
            step_size=self.sample_params['step_size'],
            noise_level=noise_level,
            extra={"conditioning": conditioning},
        )

        self.model_wrapper.train(training)
        
        return posterior_mean, posterior_var