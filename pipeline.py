import PIL
from typing import List, Optional, Union
import warnings
import torch 
import inspect

import numpy as np 
import torch.nn.functional as F

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from torch import autocast
from PIL import Image 

from utils import init_attention_weights, init_attention_edit, init_attention_func, \
                    use_last_tokens_attention, use_last_tokens_attention_weights, \
                    use_last_self_attention, save_last_tokens_attention, save_last_self_attention



from packaging import version
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.



class StableEditPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
    
    def train_text_embedding(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        text_embedding_optimization_steps: int = 4000,
        **kwargs,
    ): 
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.") 
        
        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval() 

        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone() 

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image) 

        latents_dtype = text_embeddings.dtype
        init_image = init_image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(init_image).latent_dist
        init_image_latents = init_latent_image_dist.sample(generator=generator)
        init_image_latents = 0.18215 * init_image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps))
        progress_bar.set_description("Steps")

        global_step = 0

        print("First optimizing the text embedding to better reconstruct the init image") 
        for _ in range(text_embedding_optimization_steps):
            # Sample noise that we'll add to the latents
            noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
            timesteps = torch.randint(1000, (1,), device=init_image_latents.device)
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)
            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1 
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)


        text_embeddings.requires_grad_(False)
        return (text_embeddings, text_embeddings_orig, init_image_latents)


    def train_model(
        self,
        text_embeddings,
        init_image_latents, 
        generator: Optional[torch.Generator] = None,
        diffusion_model_learning_rate: float = 2e-6,
        model_fine_tuning_optimization_steps: int = 1000,
    ):
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )

        progress_bar = tqdm(range(model_fine_tuning_optimization_steps))
        for idx in range(model_fine_tuning_optimization_steps): 
            # Sample noise that we'll add to the latents
            noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
            timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            optimizer.step() 
            optimizer.zero_grad() 

            progress_bar.update(1)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            '''
            if (idx + 1) % 500 == 0:
                image = self(
                    text_embeddings=text_embeddings,
                    generator=generator,
                ).images[0] 
                image.save('./tmp/' + str(idx+1) + '.png')
            '''
    


    @torch.no_grad()
    def __call__(
        self,
        text_embeddings,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if self.device.type == "mps":
            # randn does not exist on mps
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                self.device
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    @torch.no_grad() 
    def weighted_generate(
        self,
        prompt_emb, 
        prompt_edit_emb,
        prompt_ids,
        prompt_edit_ids, 
        prompt_edit_token_weights=[], 
        prompt_edit_tokens_start=0.0, 
        prompt_edit_tokens_end=1.0, 
        prompt_edit_spatial_start=0.0, 
        prompt_edit_spatial_end=1.0, 
        guidance_scale=7.5, 
        steps=50, 
        generator=None, 
        width=512, 
        height=512, 
        init_image=None, 
    ):

        #Set inference timesteps to scheduler
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(steps) 

        init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=self.device) 
        t_start = 0

        #Generate random normal noise
        noise = torch.randn(init_latent.shape, generator=generator, device=self.device)


        if init_image is not None: 
            noise = init_image

        init_latents = noise
        #latent = noise * scheduler.init_noise_sigma
        latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=self.device)).to(self.device)

        #Process clip
        with autocast('cuda'):
            tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = prompt_emb # self.text_encoder(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            tokens_conditional = self.tokenizer(prompt_ids, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional = prompt_emb # self.text_encoder(tokens_conditional.input_ids.to(self.device)).last_hidden_state

            #Process prompt editing
            if prompt_edit_emb is not None:
                tokens_conditional_edit = self.tokenizer(prompt_edit_ids, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional_edit = prompt_edit_emb #self.text_encoder(tokens_conditional_edit.input_ids.to(self.device)).last_hidden_state
                
                init_attention_edit(tokens_conditional, tokens_conditional_edit, self.tokenizer, self.unet, self.device)
                
            init_attention_func(unet=self.unet)
            init_attention_weights(prompt_edit_token_weights, self.tokenizer, self.unet, self.device)
                
            timesteps = scheduler.timesteps[t_start:]
            
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                #sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                #Predict the unconditional noise residual
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
                #Prepare the Cross-Attention layers
                if prompt_edit_emb is not None:
                    save_last_tokens_attention(self.unet)
                    save_last_self_attention(self.unet)
                else:
                    #Use weights on non-edited prompt when edit is None
                    use_last_tokens_attention_weights(self.unet)
                    
                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                
                #Edit the Cross-Attention layer activations
                if prompt_edit_emb is not None:
                    t_scale = t / scheduler.num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        use_last_tokens_attention(self.unet)
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        use_last_self_attention(self.unet)
                        
                    #Use weights on edited prompt
                    use_last_tokens_attention_weights(self.unet)

                    #Predict the edited conditional noise residual using the cross-attention masks
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample
                    
                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            #scale and decode the image latents with vae
            latent =  1 / 0.18215 * latent
            image = self.vae.decode(latent.to(self.vae.dtype)).sample

        '''
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        '''

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)

        return image


    @torch.no_grad() 
    def image_inversion(
            self,
            init_image, 
            prompt='', 
            prompt_emb=None, 
            guidance_scale=3.0, 
            steps=50, 
            generator=None,
            refine_iterations=3, 
            refine_strength=0.9, 
            refine_skip=0.7
        ):
            train_steps = 1000
            step_ratio = train_steps // steps
            timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(self.device)
            
            betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
            alphas = torch.cumprod(1 - betas, dim=0)
            
            init_step = 0 
            with autocast('cuda'):
                tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

                if prompt_emb is not None: 
                    embedding_conditional = prompt_emb
                else:
                    tokens_conditional = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                    embedding_conditional = self.text_encoder(tokens_conditional.input_ids.to(self.device)).last_hidden_state
        
                latent = init_image 

                for i in tqdm(range(steps), total=steps):
                    t_index = i + init_step
            
                    t = timesteps[t_index]
                    t1 = timesteps[t_index + 1]
                    #Magic number for tless taken from Narnia, used for backwards CFG correction
                    tless = t - (t1 - t) * 0.25
                    
                    ap = alphas[t] ** 0.5
                    bp = (1 - alphas[t]) ** 0.5
                    ap1 = alphas[t1] ** 0.5
                    bp1 = (1 - alphas[t1]) ** 0.5
                    
                    latent_model_input = latent
                    #Predict the unconditional noise residual
                    noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                    
                    #Predict the conditional noise residual and save the cross-attention layer activations
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                    
                    #Perform guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    #One reverse DDIM step
                    px0 = (latent_model_input - bp * noise_pred) / ap
                    latent = ap1 * px0 + bp1 * noise_pred
                    
                    #Initialize loop variables
                    latent_refine = latent
                    latent_orig = latent_model_input
                    min_error = 1e10
                    lr = refine_strength

                    #Finite difference gradient descent method to correct for classifier free guidance, performs best when CFG is high
                    #Very slow and unoptimized, might be able to use Newton's method or some other multidimensional root finding method
                    if i > (steps * refine_skip):
                        for k in range(refine_iterations):
                            #Compute reverse diffusion process to get better prediction for noise at t+1
                            #tless and t are used instead of the "numerically correct" t+1, produces way better results in practice, reason unknown...
                            noise_pred_uncond = self.unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
                            noise_pred_cond = self.unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            
                            #One forward DDIM Step
                            px0 = (latent_refine - bp1 * noise_pred) / ap1
                            latent_refine_orig = ap * px0 + bp * noise_pred
                            
                            #Save latent if error is smaller
                            error = float((latent_orig - latent_refine_orig).abs_().sum())
                            if error < min_error:
                                latent = latent_refine
                                min_error = error

                            #print(k, error)
                            
                            #Break to avoid "overfitting", too low error does not produce good results in practice, why?
                            if min_error < 5:
                                break
                            
                            #"Learning rate" decay if error decrease is too small or negative (dampens oscillations)
                            if (min_error - error) < 1:
                                lr *= 0.9

                            #Finite difference gradient descent
                            latent_refine = latent_refine + (latent_model_input - latent_refine_orig) * lr 

            return latent 
