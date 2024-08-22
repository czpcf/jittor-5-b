# 第四届计图人工智能挑战赛：赛题二总结

![](https://github.com/czpcf/jittor-5-b/blob/main/pic.png?raw=true)

本项目参考自 [HuggingFace的训练指导](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora) 。

## 权重下载

https://cloud.tsinghua.edu.cn/f/c4a45eea8c2f4a6ebd25/?dl=1

解压至jittor-5-b/JDiffusion_singlepack，新建文件夹style并放于其中

## 环境安装

### 安装步骤

首先配置虚拟环境：

```bash
conda create -n jdiffusion_modified python=3.9
conda activate jdiffusion_modified
```

安装库：

```bash
python -m pip install git+https://github.com/JittorRepos/jittor
python -m pip install git+https://github.com/JittorRepos/transformers_jittor
python -m pip install git+https://github.com/JittorRepos/jtorch
```

接下来进入`how`目录下：

```bash
pip install -e diffusers_jittor
pip install -e peft-0.10.0
pip install -e JDiffusion
```

最后调整到合适的accelerate版本：

```bash
pip install accelerate==0.27.2
```

### 特别说明

修改的文件包括：
/peft-0.10.0/src/peft/tuners/tuners_utils.py
/diffusers_jittor/src/diffusers/loaders/lora.py
/diffusers_jittor/src/diffusers/loaders/unet.py
/diffusers_jittor/src/diffusers/models/modeling_utils.py

需要说明的是，peft-0.10.0中的adapter与jdiffusion不能很好的兼容，使用原版peft会导致jdiffusion的训练结果（unet的to_out.0权重）不能正确地注入到底模中。此处仅仅修改了读入lora权重的路径，属于是修复了bug。




如果遇到问题No module named 'cupy'，尝试：

```
# Install CuPy from source
pip install cupy
# Install CuPy for cuda11.2 (Recommended, change cuda version you use)
pip install cupy-cuda112
```

## 训练

训练的文件在 `JDiffusion_singlepack` 中。对应的B题的数据集已经放在文件夹里。

在命令行中输入 `bash train_joint_pack3_3.sh` 即可开始训练。首次运行可能失败，重新开始即可。在1张3090上需要的时间约为5个小时。

需要注意，由于每4000步会保存权重约为1.18个G的文件，请至少预留15G的硬盘空间。

如果只需要最后的结果，修改 `train_joint_pack3_3.sh` 中第22行的 `SAVE_FREQUENCY` 为40000即可。

## 推理

在命令行中输入 `python run_joint_pack3_3_parallel.py` 即可。

## 训练方案与结果

### 1. 给底模的transformer增加更多的lora（成功）+文本描述

该方案主要是给transformer的to_k, to_q, to_v, to_out, add_k_proj, add_v_proj增加LoRA；给text_encoder的q_proj, k_proj, v_proj增加LoRA。

由于unet在训练的时候，还会将输入图片对应的文本经过text_encoder变成隐空间的向量，而这一部分原数据集只有总体的描述，没有更细致内容。因此还对原数据集的每一张图片手动标注了内容，增加了0到4个prompt，这些prompt一部分描述具体物体，一部分描述风格共性。这一部分其实可以被CLIP/BLIP代替。

在A榜时，同参数量的情况下，使用纯baseline训练500步分数为0.22，而加上这些LoRA训练500步能达到0.43的分数，再加上额外的文本描述可以达到0.48的分数。

### 2. adaptive lr（成功）

参考文献[2]指出，LoRA的B矩阵的学习率需要大于A矩阵的学习率，并且量级上大约为 $O(\sqrt{\frac{n}{r}})$ 的级别。在训练中，B矩阵的学习率被设置为 $\gamma\sqrt{\frac{n}{r}}$ ，其中 $\gamma$ 为一超参， $n$ 为LoRA对应的维度（取值在[320,640,1024,1280]中）。在B榜上有4次提交，其分数如下：
```plain
训练步数10000的baseline+adaptive:0.4251
训练步数10000的baseline         :0.4243
训练步数16000的baseline+adaptive:0.4406
训练步数16000的baseline         :0.4359
```
可见同步数时加入adaptive lr仍有一定作用。


### 3. LoRA-GA（失败）

参考文献[3]指出，给LoRA选取一批样本，计算初始梯度$G_0=\nabla_{W_0}\mathcal{L}$，对梯度进行SVD分解为$G_0=U\Sigma V$，取$U$的前$r$列初始化$A$，取$V$的第$r+1\sim 2r$行初始化$B$能达到更好得到效果。

然而最后的推理结果在视觉上就很不成功。因此没有尝试提交。


### 4. 打包训练（成功）

我们注意到如果只使用一种风格的少量图片来微调，模型容易把具体物体的特征学习进风格特征中去。为此我们认为应当延续dreambooth的思路，通过设计来保持模型生成不同种类物体的能力。

起初的尝试，是类似dreambooth的：用模型自己一开始对物体给出的图片来监督后续训练过程，让模型在没有trigger的时候能生成和最开始类似的图片。但是我们注意到模型直接生成图片质量不高，影响微调质量。更进一步，这样训练速度较慢。

为此我们决定将不同风格的图片放在一起微调，这样涉及的物体种类更多，可以一定程度上保持模型能力避免遗忘或者风格对物体信息的覆盖，而且不同风格一起微调，不会太影响训练效率。

技术上，在打包中，如果一个包有n个style，打包前每个style的lora的rank为r，则打包后lora的rank为n*r。这是如果使用2 adaptive lr，则可能导致B矩阵的学习率小于A。对此我们的措施是，让B相对A的学习率倍率与1取max。

我们尝试了将相近的风格打包，或者将相异的风格打包。但是我们最终发现，将所有图片放在一起训练，整体效果最好，速度最快。

事实上该思路也能用于单个风格训练，只需要额外准备若干高质量、种类丰富的图片，将其打包为若干虚拟风格即可，只是训练效率就会较低。

见下面数据（n每翻一倍，训练时间也翻一倍）：

```
n=2
dino_score：0.2125
hist_score：0.747
fid_score：2.813
lpips_score：0.3939
clip_iqa_score：0.7451
clip_r_score：0.62
total_score：0.4251

n=8
dino_score：0.1916
hist_score：0.73
fid_score：3.8127
lpips_score：0.4113
clip_iqa_score：0.7579
clip_r_score：0.7629
total_score：0.4419

n=14
dino_score：0.1762
hist_score：0.6811
fid_score：3.1457
lpips_score：0.4223
clip_iqa_score：0.7471
clip_r_score：0.8457
total_score：0.4497

n=28
dino_score：0.1737
hist_score：0.7144
fid_score：3.272
lpips_score：0.4154
clip_iqa_score：0.7463
clip_r_score：0.8586
total_score：0.4648
```

可见n越大相应的r_clip_score会变得更大，但是dino_score会下降。

### 5. 使用梯度下降优化风格描述hard prompt

我们尝试使用一些prompt engineering的方法来得到较优的风格描述，称之为style prompt，随后我们将这些style prompt和trigger word(即style_xx)放在一起作为整个trigger，开展后续微调与推理。

该方法的动机是，有很多仅针对text encoder甚至仅仅优化prompt的工作，已经能实现对图像的较好控制。这里先使用优化算法来得到初始prompt，一方面能降低后续微调难度，另一方面能避免人工标注的偏差。

具体代码参考了参考文献[4]。大体上，要得到描述某个风格的style prompt，我们先对同一个风格的图片给出关于其物体的tag，接下来的目标就是让style prompt+tag和相应图片在经过CLIP后得到的特征向量尽可能相似。为此我们style prompt通过latent embedding的形式存储为soft prompt，每次将每个token映射到最近的对应实际词汇的token上，得到hard prompt。计算得到hard prompt的优化梯度，接下来将该梯度作用于soft prompt上。经过映射后优化效果较好，而且得到的hard prompt可以用于后续训练推理。

由于每个风格图片较少，为避免这样学习得到的style prompt含有具体物体信息，在loss中还加入了一项为任取表示实际物体的prompt P，要求style prompt+P和P在经过CLIP后得到的embedding尽量接近。

使用该方法后，可以在风格学习效果上得到较好效果，但是我们没有来得及探索出最合适的超参数，因此最终最优版本未使用。

### 6. 用大预言模型生成描述（成功）

一开始生成的时候仅仅是使用下发的prompt，这样做clip_r_score会比较低。通过增加LLM生成的上一级描述能够较为显著地增加clip_r_score，从而提高分数。

### 7. 加negative prompt（失败）

这个方法是指在推理的时候，加入一些与positive prompt看上去不相关的negative prompt来“更好”地引导梯度流。然而在A榜中的每一次提交，最后的分数都会变差，而且每一次的clip_r_score和dino_score都减少了。B榜中的两次提交如下：

```
只有negative prompt差别，其余参数均一致（10000步+baseline+adaptive）:

没negative：
dino_score：0.2078
hist_score：0.7663
fid_score：3.8347
lpips_score：0.3976
clip_iqa_score：0.7459
clip_r_score：0.6843
total_score：0.4359

有negative：
dino_score：0.1921
hist_score：0.7456
fid_score：3.3376
lpips_score：0.4114
clip_iqa_score：0.7591
clip_r_score：0.6771
total_score：0.4278
```

## 最终的训练方案与结果
最后训练时使用了1+2+4+6的策略。在一个3090上训练用时约5个小时。

最后该训练方式的结果如下：
```
dino_score：0.1737
hist_score：0.7144
fid_score：3.272
lpips_score：0.4154
clip_iqa_score：0.7463
clip_r_score：0.8586
total_score：0.4648
```
可见fid实在是拉跨（悲）

# LLM的prompt

训练的llm prompt由ChatGLM生成，询问prompt如下：
```
You are an expert at stable diffusion prompt engineering, and you are engaged in a task about providing both high quality positive and negative prompts to generate delicate images. Your task is to generate three types of prompts given input word:synonyms but same, synonyms but different, and hypernyms.

The process is as follows:

1. An input word is given in one single line, the spaces will be replaced with underline '_'.
2. provide as many words as possible in three catagories:different words, property, and hypernyms. You should note that different words mean that they are different in the same class, for example: 'cat' and 'dog' are same words under the word 'animal', so they are differnt words, but 'cat' and 'watermelon' are not same words under the word 'animal'.Property describes what the prompt may have, it can be concrete or abstract. And hypernyms indicates that the prompt is a class(or special case) of it.
3. return your reply in json format. Spaces MUST be replaced with underline '_', for example: 'dark background' should be 'dark_background'.

For example:
> input: cat
> return:
{
	"prompt":"cat",
	"different":["dog","horse","pig"],
	"property":["fur","paws","tail","cute"],
	"hypernyms":["animal","creature","pet"]
}

Another example:
> input: sand
> return:
{
	"prompt":"sand"
	"different":["dirt","grass","rock"],
	"property":["particle","hard"],
	"hypernyms":["material","substance"]
}

If you understand, say 'yes' and I will give you inputs and you should NEVER do anything else.
```


## 参考文献

```
[1]
@inproceedings{ruiz2023dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

[2]
@online{kexuefm-10001,
        title={配置不同的学习率，LoRA还能再涨一点？},
        author={苏剑林},
        year={2024},
        month={Feb},
        url={\url{https://spaces.ac.cn/archives/10001}},
}

[3]
@online{kexuefm-10226,
        title={对齐全量微调！这是我看过最精彩的LoRA改进（一）},
        author={苏剑林},
        year={2024},
        month={Jul},
        url={\url{https://spaces.ac.cn/archives/10226}},
}

[4]
@misc{wen2023hardpromptseasygradientbased,
      title={Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery}, 
      author={Yuxin Wen and Neel Jain and John Kirchenbauer and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2023},
      eprint={2302.03668},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2302.03668}, 
}
```
