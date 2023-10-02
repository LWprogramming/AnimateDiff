# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.loaders import TextualInversionLoaderMixin

from einops import rearrange

from ..models.unet import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


#### This code is from animatediff prompt travel repo, originally in context.py but moved here for simplicity.
# Used in uniform.
def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)
    return as_int / (1 << 64)

# This takes some information about the denoising process and returns a list of integers corresponding to the indices of the frames we want to pull out.
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]

# Just returns uniform, since nothing else is implemented.
def get_context_scheduler(name: str) -> Callable:
    match name:
        case "uniform":
            return uniform
        case _:
            raise ValueError(f"Unknown context_overlap policy {name}")


# interpolation methods
def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()

#### End of animatediff copied code.

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
    #     batch_size = len(prompt) if isinstance(prompt, list) else 1

    #     text_inputs = self.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=self.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     text_input_ids = text_inputs.input_ids
    #     untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    #     if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
    #         removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
    #         logger.warning(
    #             "The following part of your input was truncated because CLIP can only handle sequences up to"
    #             f" {self.tokenizer.model_max_length} tokens: {removed_text}"
    #         )

    #     if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
    #         attention_mask = text_inputs.attention_mask.to(device)
    #     else:
    #         attention_mask = None

    #     text_embeddings = self.text_encoder(
    #         text_input_ids.to(device),
    #         attention_mask=attention_mask,
    #     )
    #     text_embeddings = text_embeddings[0]

    #     # duplicate text embeddings for each generation per prompt, using mps friendly method
    #     bs_embed, seq_len, _ = text_embeddings.shape
    #     text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
    #     text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    #     # get unconditional embeddings for classifier free guidance
    #     if do_classifier_free_guidance:
    #         uncond_tokens: List[str]
    #         if negative_prompt is None:
    #             uncond_tokens = [""] * batch_size
    #         elif type(prompt) is not type(negative_prompt):
    #             raise TypeError(
    #                 f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
    #                 f" {type(prompt)}."
    #             )
    #         elif isinstance(negative_prompt, str):
    #             uncond_tokens = [negative_prompt]
    #         elif batch_size != len(negative_prompt):
    #             raise ValueError(
    #                 f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
    #                 f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
    #                 " the batch size of `prompt`."
    #             )
    #         else:
    #             uncond_tokens = negative_prompt

    #         max_length = text_input_ids.shape[-1]
    #         uncond_input = self.tokenizer(
    #             uncond_tokens,
    #             padding="max_length",
    #             max_length=max_length,
    #             truncation=True,
    #             return_tensors="pt",
    #         )

    #         if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
    #             attention_mask = uncond_input.attention_mask.to(device)
    #         else:
    #             attention_mask = None

    #         uncond_embeddings = self.text_encoder(
    #             uncond_input.input_ids.to(device),
    #             attention_mask=attention_mask,
    #         )
    #         uncond_embeddings = uncond_embeddings[0]

    #         # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    #         seq_len = uncond_embeddings.shape[1]
    #         uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
    #         uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

    #         # For classifier free guidance, we need to do two forward passes.
    #         # Here we concatenate the unconditional and text embeddings into a single batch
    #         # to avoid doing two forward passes
    #         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    #     return text_embeddings

    # I don't really understand what this is doing, but this is overwriting the original AnimateDiff repo's _encode_prompt method. I copy-pasted this directly from the prompt-travel repo: https://github.com/s9roll7/animatediff-cli-prompt-travel/blob/f67dfdc138b93dce96e5e32d7b3e932d3ab7a3f5/src/animatediff/pipelines/animation.py#L247, although things aren't super clear based on issue https://github.com/s9roll7/animatediff-cli-prompt-travel/issues/83.
    # I also had to download utils/lpw_stable_diffusion.py from the prompt-travel repo and the models/clip.py file to make things work because there's a pile of code to add there.
    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: int = 1,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        from ..utils.lpw_stable_diffusion import get_weighted_text_embeddings

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_map: Dict[int, str] = None,
        context_frames: int = -1,
        context_stride: int = 3,
        context_overlap: int = 4,
        **kwargs,
    ):
        assert context_frames > 0, "context_frames must be greater than 0" # need to set this to something other than -1 by hand; normally i'd make this mandatory arg but i do it like this to match how it's done in animatediff prompt travel repo
        # TODO: turns out removing prompt would take a lot of work so we're going to keep it in and just not use it
        # example prompt map:
        # prompt_map = {
        #     0: "A person is walking",
        #     10: "A person is running",
        #     20: "A person is jumping",
        #     30: "A person is dancing",
        # }
        # # prompt map is for multi prompt generation
        # assert prompt is None or prompt_map is None, "Only one of `prompt` and `prompt_map` can be set."

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        # text_embeddings = self._encode_prompt(
        #     prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        # )

        # we're just copying code mostly, out of https://github.com/s9roll7/animatediff-cli-prompt-travel/blob/f67dfdc138b93dce96e5e32d7b3e932d3ab7a3f5/src/animatediff/pipelines/animation.py#L64
        # if using a prompt_map then we need to encode all the prompts
        prompt_embeds_map = {}
        prompt_map = dict(sorted(prompt_map.items()))
        prompt_list = [prompt_map[key_frame] for key_frame in prompt_map.keys()]
        # TODO: this _encode_prompt api has probably changed
        prompt_embeds = self._encode_prompt(
            prompt_list,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            # clip_skip=clip_skip,
        )
        if do_classifier_free_guidance:
            negative, positive = prompt_embeds.chunk(2, 0)
            negative = negative.chunk(negative.shape[0], 0)
            positive = positive.chunk(positive.shape[0], 0)
        else:
            positive = prompt_embeds
            positive = positive.chunk(positive.shape[0], 0)
        for i, key_frame in enumerate(prompt_map):
            if do_classifier_free_guidance:
                prompt_embeds_map[key_frame] = torch.cat([negative[i] , positive[i]])
            else:
                prompt_embeds_map[key_frame] = positive[i]
        key_first =list(prompt_map.keys())[0]
        key_last =list(prompt_map.keys())[-1]
        def get_current_prompt_embeds_from_text(
                center_frame = None,
                video_length : int = 0
                ):
            # finds the nearest key frames (before and after) and just blends their prompt embeddings with slerp (sphere linear interpolation)

            key_prev = key_last
            key_next = key_first

            for p in prompt_map.keys():
                if p > center_frame:
                    key_next = p
                    break
                key_prev = p

            dist_prev = center_frame - key_prev
            if dist_prev < 0:
                dist_prev += video_length
            dist_next = key_next - center_frame
            if dist_next < 0:
                dist_next += video_length

            if key_prev == key_next or dist_prev + dist_next == 0:
                return prompt_embeds_map[key_prev]

            rate = dist_prev / (dist_prev + dist_next)

            # TODO: there's also linear but we'll pick one for now
            return slerp( prompt_embeds_map[key_prev], prompt_embeds_map[key_next], rate ) 

        def get_current_prompt_embeds_multi(
                context: List[int] = None,
                video_length : int = 0
                ):
            # Takes list of frames to use for context and returns the prompt embeddings for those frames. Skipping image-based prompts for now.

            # multi is because in the animatediff prompt travel repo, they have a multi prompt mode where they use multiple prompts and a single where there's a single prompt. But for prompt travel of course we're going to use multiple prompts.

            neg = []
            pos = []
            for c in context:
                t = get_current_prompt_embeds_from_text(c, video_length)
                if do_classifier_free_guidance:
                    negative, positive = t.chunk(2, 0)
                    neg.append(negative)
                    pos.append(positive)
                else:
                    pos.append(t)

            if do_classifier_free_guidance:
                neg = torch.cat(neg)
                pos = torch.cat(pos)
                text_emb = torch.cat([neg , pos])
            else:
                pos = torch.cat(pos)
                text_emb = pos

            # assert self.ip_adapter is None
            return text_emb
            # if self.ip_adapter == None:
                # return text_emb

            # neg = []
            # pos = []
            # for c in context:
            #     im = get_current_prompt_embeds_from_image(c, video_length)
            #     if do_classifier_free_guidance:
            #         negative, positive = im.chunk(2, 0)
            #         neg.append(negative)
            #         pos.append(positive)
            #     else:
            #         pos.append(im)

            # if do_classifier_free_guidance:
            #     neg = torch.cat(neg)
            #     pos = torch.cat(pos)
            #     image_emb = torch.cat([neg , pos])
            # else:
            #     pos = torch.cat(pos)
            #     image_emb = pos

            # return torch.cat([text_emb,image_emb], dim=1)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            # text_embeddings.dtype,
            prompt_embeds.dtype, # switched to prompt_embeds because we're doing multi prompt work now
            device,
            generator,
            latents,
        )
        # latents shape: (batch_size * num_videos_per_prompt, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # so we see latents.shape[2] a lot -- that's the video length

        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        # The key difference between original animatediff and current animatediff is that the original one uses a single prompt and the current one uses multiple prompts. We have a context scheduler that pulls out the frames we want, so the noise prediction is kind of like a blend of the noise predictions for the different prompts.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        context_scheduler = get_context_scheduler("uniform") # this is just hardcoded, from animatediff prompt travel repo
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # unlike original animatediff code, we're accumulating the noise predictions for the different prompts instead of one-shotting it like in the commented out code: noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)
                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                # counter is to keep track of how many times we've added to the noise_pred, so we can average it later
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                # # predict the noise residual
                # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)
                # # noise_pred = []
                # # import pdb
                # # pdb.set_trace()
                # # for batch_idx in range(latent_model_input.shape[0]):
                # #     noise_pred_single = self.unet(latent_model_input[batch_idx:batch_idx+1], t, encoder_hidden_states=text_embeddings[batch_idx:batch_idx+1]).sample.to(dtype=latents_dtype)
                # #     noise_pred.append(noise_pred_single)
                # # noise_pred = torch.cat(noise_pred)

                # this is the part that's different from the original animatediff code-- this is from animatediff prompt travel repo.
                # We're pulling out the frames we want to use for context. Commenting out controlnet and reference stuff for now, which are image-based or similar ways of additional conditions.
                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):
                    # if controlnet_image_map:
                        # controlnet_target = list(range(context[0]-context_frames, context[0])) + context + list(range(context[-1]+1, context[-1]+1+context_frames))
                        # controlnet_target = [f%video_length for f in controlnet_target]
                        # controlnet_target = list(set(controlnet_target))

                        # process_controlnet(controlnet_target)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents[:, :, context]
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    ) # remember, latents shape is (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor). So this pulls out the frames we want to use for context and then repeats them twice if we're doing classifier free guidance.
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # cur_prompt = get_current_prompt_embeds(context, latents.shape[2])
                    cur_prompt = get_current_prompt_embeds_multi(context, latents.shape[2]) # the animatediff prompt travel repo has a single prompt mode and a multi prompt mode. We're using multi prompt mode here, so we just call the multi prompt function.

                    # down_block_res_samples,mid_block_res_sample = get_controlnet_result(context)

                    # if c_ref_enable:
                    #     # ref only part
                    #     noise = randn_tensor(
                    #         ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                    #     )
                    #     ref_xt = self.scheduler.add_noise(
                    #         ref_image_latents,
                    #         noise,
                    #         t.reshape(
                    #             1,
                    #         ),
                    #     )
                    #     ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                    #     stopwatch_record("C_REF_MODE write start")
                    #     C_REF_MODE = "write"
                    #     self.unet(
                    #         ref_xt,
                    #         t,
                    #         encoder_hidden_states=cur_prompt,
                    #         cross_attention_kwargs=cross_attention_kwargs,
                    #         return_dict=False,
                    #     )
                    #     stopwatch_record("C_REF_MODE write end")
                    #     C_REF_MODE = "read"

                    # predict the noise residual

                    # stopwatch_record("normal unet start")
                    pred = self.unet(
                        latent_model_input.to(self.unet.device, self.unet.dtype),
                        t,
                        encoder_hidden_states=cur_prompt,
                        # cross_attention_kwargs=cross_attention_kwargs,
                        # down_block_additional_residuals=down_block_res_samples,
                        # mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0] # only care about the commented out args for controlnet: cross_attention_kwargs, down_block_additional_residuals, mid_block_additional_residual

                    # stopwatch_record("normal unet end")

                    pred = pred.to(dtype=latents.dtype, device=latents.device)
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()

                # perform guidance
                if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
