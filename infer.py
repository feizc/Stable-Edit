import torch 
from PIL import Image 
import os 

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from pipeline import StableEditPipeline 


def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda') 

    model_path = './ckpt'
    generator = torch.Generator(device).manual_seed(2022)

    # save for tempt files
    tmp_path = './tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    init_image = Image.open('woman.png').convert('RGB') 
    init_image = init_image.resize((512, 512)) 

    prompt = 'a photo of a young woman with wavy black hair, smiling' 
    original_prompt = 'a photo of a young woman'

    pipe = StableEditPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
    ).to(device) 

    text_train_step = 4000
    model_train_step = 5000

    # optimize text embedding
    text_emb, text_emb_ori, image_latents = pipe.train_text_embedding(
            prompt=prompt,
            init_image=init_image,
            generator=generator,
            text_embedding_optimization_steps=text_train_step,
        )
    image = pipe(
            text_embeddings=text_emb,
            generator=generator,
        ).images[0] 
    image.save('./tmp/text_emb.png')

    # optimize u-net
    pipe.train_model(
        text_embeddings=text_emb,
        init_image_latents=image_latents, 
        generator=generator,
        model_fine_tuning_optimization_steps=model_train_step,
    )

    image = pipe(
        text_embeddings=text_emb,
        generator=generator,
        ).images[0] 
    image.save('./tmp/unet.png')

    # get image latent 
    image_latents = pipe.image_inversion(
        init_image=image_latents,
        prompt=original_prompt
    )

    scalr = 0.0
    for i in range(11): 
        image = pipe.weighted_generate(
            prompt_emb=text_emb, 
            prompt_ids=original_prompt, 
            prompt_edit_emb=text_emb_ori,
            prompt_edit_ids=prompt,
            generator=generator,
            prompt_edit_spatial_start=scalr 
        )
        image.save('./tmp/edit_text_' + str(i) +'.png')
        scalr += 0.1 

    scalr = 0.0 
    for i in range(11): 
        image = pipe.weighted_generate(
            prompt_emb=text_emb, 
            prompt_ids=original_prompt, 
            prompt_edit_emb=text_emb_ori,
            prompt_edit_ids=prompt,
            generator=generator,
            init_image=image_latents,
            init_image_strength=1.0 - scalr,
        )
        image.save('./tmp/edit_img_' + str(i) + '.png')
        scalr += 0.1 


if __name__ == '__main__': 
    main()