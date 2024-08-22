import json, os, tqdm
import jittor as jt
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from JDiffusion.pipelines import StableDiffusionPipeline

min_num = 0
max_num = 27
dataset_root = "B_dataset"

model_name = "joint_pack3_3"

step_min = 40000
step_max = 40000
step_gap = 4000

skip_when_exits = True

what_model = [
    "00", #00
    "00", #01
    "00", #02
    "00", #03
    "00", #04
    "00", #05
    "00", #06
    "00", #07
    "00", #08
    "00", #09
    "00", #10
    "00", #11
    "00", #12
    "00", #13
    "00", #14
    "00", #15
    "00", #16
    "00", #17
    "00", #18
    "00", #19
    "00", #20
    "00", #21
    "00", #22
    "00", #23
    "00", #24
    "00", #25
    "00", #26
    "00", #27
]

positive = [
    "purple, lineart, dark_background,", # 00
    "watercolor, paint,", # 01
    "yarn, wool,", # 02
    "lineart, clean,", # 03
    "cloud, sky,", # 04
    "gray_background", #05
    "paper cut,red,", #06
    "street,", # 07
    "paperart,", # 08
    "watercolor,", # 09
    "radial,stripes,", # 10
    "simple_background, heavy_line,", # 11
    "blocks,", # 12
    "chinese_waterpaint,", # 13
    "simple_background, lineart,", # 14
    "colorful,", # 15
    "pixelart,flat,2d,", # 16
    "waterpaint,", # 17
    "cute,", # 18
    "blocks, pixelart,building_block,", # 19
    "sketch", # 20
    "paper cut,depth_of_field,", # 21
    "pencil_art", # 22
    "simple_background,", # 23
    "rubber,", # 24
    "pixel,bricks,", # 25
    "paperfold,", # 26
    "purple_background", # 27
]

repeat_times = [
    4, # 00
    4, # 01
    4, # 02
    4, # 03
    4, # 04
    4, # 05
    4, # 06
    4, # 07
    4, # 08
    4, # 09
    4, # 10
    4, # 11
    4, # 12
    4, # 13
    4, # 14
    4, # 15
    4, # 16
    4, # 17
    4, # 18
    4, # 19
    4, # 20
    4, # 21
    4, # 22
    4, # 23
    4, # 24
    4, # 25
    4, # 26
    4, # 27
]

use_llm_prompt = [
    True, # 00
    True, # 01
    True, # 02
    True, # 03
    True, # 04
    True, # 05
    True, # 06
    True, # 07
    True, # 08
    True, # 09
    True, # 10
    True, # 11
    True, # 12
    True, # 13
    True, # 14
    True, # 15
    True, # 16
    True, # 17
    True, # 18
    True, # 19
    True, # 20
    True, # 21
    True, # 22
    True, # 23
    True, # 24
    True, # 25
    True, # 26
    True, # 27
]

llm_prompt = open(f"./{dataset_root}/llm_prompt.json")
llm_prompt = json.load(llm_prompt)
max_pack = 8

with jt.no_grad():
    for weight_suf in range(step_min, step_max + step_gap, step_gap):
        for tempid in tqdm.tqdm(range(min_num, max_num+1)):
            taskid = "{:0>2d}".format(tempid)
            print(f"style/style_{taskid}_{model_name}_{weight_suf}")
            model_loaded = False

            # load llm_prompt
            with open(f"{dataset_root}/llm_prompt.json", "r") as file:
                llm_prompts = json.load(file)
        
            # load json
            with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
                prompts = json.load(file)

            pack_pos = []
            pack_neg = []
            pack_dir = []
            pack_path = []
            for id, prompt in prompts.items():
                image_dir = f"./output_{model_name}_continous/{weight_suf}/{taskid}"
                image_path = os.path.join(image_dir, f"{prompt}.png")

                if skip_when_exits and os.path.exists(image_path): 
                    continue

                if not model_loaded:
                    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to('cuda')
                    print("base loaded")
                    pipe.load_lora_weights(f"style/style_{what_model[tempid]}_{model_name}_{weight_suf}")
                    print("lora loaded")
                    model_loaded = True

                # pos_sentence = "A drawing of "
                pos_sentence = ""
                pos_sentence += f"style_{taskid},"+f"{positive[tempid]}"+(prompt+",")*repeat_times[tempid]
                
                neg_sentence = 'worst quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg, artifacts, signature, username, error, sketch, duplicate, ugly, horror'
                if use_llm_prompt[tempid] and prompt in llm_prompt:
                    
                    if llm_prompt[prompt] != None:
                        for v in llm_prompt[prompt]['different']:
                            neg_sentence += "," + v
                        for v in llm_prompt[prompt]['hypernyms']:
                            pos_sentence += "," + v
                    else:
                        print("WARNING: prompt {} has no llm_prompts!".format(prompt))
                
                pos_sentence = pos_sentence.replace('  ',' ')
                neg_sentence = neg_sentence.replace('  ',' ')
                pos_sentence = pos_sentence.replace(',,',',')
                neg_sentence = neg_sentence.replace(',,',',')
                pos_sentence = pos_sentence.replace(', ',',')
                neg_sentence = neg_sentence.replace(', ',',')
                print("positive:",pos_sentence)
                print("negative:",neg_sentence)
                
                # use negative prompt
                # image = pipe(prompt=pos_sentence, negative_prompt=neg_sentence, num_inference_steps=50, width=512, height=512, clip_skip=0, seed=114514, guidance_scale=7.5).images[0]
                
                pack_pos.append(pos_sentence)
                pack_neg.append(neg_sentence)
                pack_dir.append(image_dir)
                pack_path.append(image_path)
                
                if len(pack_pos) == max_pack:
                    images = pipe(prompt=pack_pos, num_inference_steps=100, width=512, height=512, clip_skip=0, seed=114514, guidance_scale=7.5).images
                    for i in range(max_pack):
                        image = images[i]
                        os.makedirs(pack_dir[i], exist_ok=True)
                        image.save(pack_path[i])
                    pack_pos.clear()
                    pack_neg.clear()
                    pack_dir.clear()
                    pack_path.clear()
            
            if len(pack_pos) != 0:
                images = pipe(prompt=pack_pos, num_inference_steps=100, width=512, height=512, clip_skip=0, seed=114514, guidance_scale=7.5).images
                for i in range(len(pack_pos)):
                    image = images[i]
                    os.makedirs(pack_dir[i], exist_ok=True)
                    image.save(pack_path[i])
                # os.makedirs(f"./output_{model_name}_no_adaptive_newpack_10000/{taskid}", exist_ok=True)
                # image.save(f"./output_{model_name}_no_adaptive_newpack_10000/{taskid}/{prompt}.png")