# Stable-Edit 

This is the PyTorch implementation for image editing framework as described in: 

> **Stable-Edit: Text-based real image editing with stable diffusion models**

We address the consistency image editing by inversing both image and text embedding. 


## 1. Model Structure 

<p align="center">
     <img src="figures/framework.png" alt="Stable edit framework" width = "600">
     <br/>
     <sub><em>
     Overview of the proposed stable editing framework.
    </em></sub>
</p>

Specifically, provide with the input image $X$ and target text $Y$, we first learn the inversed text embedding $e_{opt}$ of image $X$. Then, we combine the target text embedding $e_{tgt}$ from text encoder and inversed text embedding $e_{opt}$ with cross attention. Next, we learn the inversed image latents according to image latents $h_{in}$ of image $X$ from VAE according to DDIM scheduler. Finally, we forward the standard text-to-image generation.  


## 2. Cases

<p align="center">
     <img src="figures/case.png" alt="Edited cases">
     <br/>
     <sub><em>
     Cases for the image editing.
    </em></sub>
</p>



This repository is based on [diffusers](https://github.com/huggingface/diffusers).
