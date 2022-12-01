"""
input: a sequece of sentence
output: a sequence of image 
"""
import torch 
import os 

from pipeline import StableEditPipeline 



def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda') 

    model_path = './ckpt'
    generator = torch.Generator(device).manual_seed(2022) 

    # save for result files
    tmp_path = './tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    pipe = StableEditPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
    ).to(device) 

    prompt_list = ["a cat sitting on a grass", "a cat standing on a grass",  "a cat running on a grass", ] 
    
    # iteration for image generation
    for i in range(len(prompt_list)): 
        if i == 0: 
            image = pipe.prompt_generate(
                prompt_ids=prompt_list[0],
                generator=generator,
            )[0] 
            image.save(os.path.join(tmp_path, '1.png')) 
        else:
            prompt_edit_spatial_end = 0.0 
            for j in range(11): 
                image = pipe.prompt_generate(
                    prompt_ids=prompt_list[i-1],
                    prompt_edit_ids=prompt_list[i],
                    generator=generator, 
                    prompt_edit_spatial_end=prompt_edit_spatial_end,
                )[0]
                prompt_edit_spatial_end += 0.1 
                image.save(os.path.join(tmp_path, str(i)+'_'+str(j)+'.png'))




if __name__ == '__main__': 
    main() 
