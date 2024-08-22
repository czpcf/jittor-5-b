#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

model_name = 'joint_pack3_3'

from datetime import datetime

import jittor as jt
import jittor.nn as nn
import torch
import argparse
import copy
import logging
import math
import os
import warnings
from pathlib import Path
from peft import get_peft_model
import json
import random

import numpy as np
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from JDiffusion import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers import DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0] # CLIPTextModel

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_process",
        type=int,
        default=1
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pack",
        type=str,
        default="",
        help="Pack.",
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        default="loss",
        help="The output directory where the loss will be written.",
    )
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=5, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=1000,
        help=("Save frequency."),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help=("Dropout."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # logger is not available yet
    if args.class_data_dir is not None:
        warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    if args.class_prompt is not None:
        warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    
    def set_order(self):
        order = []
        for style in self.styles:
            for i in range(len(self.instance_images_path[style])):
                order.append([style, i])
        np.random.shuffle(order)
        self.order = order

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        pack=None,
    ):
        self.styles = pack.split(',')
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = {}
        self.instance_images_path = {}
        self.num_instance_images = {}
        self.instance_prompt = {}
        self._length = {}
        
        for style in self.styles:
            
            self.instance_data_root[style] = Path(instance_data_root+f"/{style}/images")
            print(self.instance_data_root[style])
            if not self.instance_data_root[style].exists():
                raise ValueError("Instance images root doesn't exists.")
            
            self.instance_images_path[style] = list(Path(self.instance_data_root[style]).iterdir())
            self.num_instance_images[style] = len(self.instance_images_path[style])
            self.instance_prompt[style] = instance_prompt + f'{style}'
            self._length = len(self.instance_images_path[style])

            if class_data_root is not None:
                assert(0)
                # self.class_data_root = Path(class_data_root)
                # self.class_data_root.mkdir(parents=True, exist_ok=True)
                # self.class_images_path = list(self.class_data_root.iterdir())
                # if class_num is not None:
                #     self.num_class_images = min(len(self.class_images_path), class_num)
                # else:
                #     self.num_class_images = len(self.class_images_path)
                # self._length = max(self.num_class_images, self.num_instance_images)
                # self.class_prompt = class_prompt
            else:
                self.class_data_root = None

        self.image_transforms_no_flip = transform.Compose(
            [
                transform.Resize(size),
                transform.CenterCrop(size) if center_crop else transform.RandomCrop(size),
                transform.ToTensor(),
                transform.ImageNormalize([0.5], [0.5]),
            ]
        )
        
        self.image_transforms_flip = transform.Compose(
            [
                transform.Resize(size),
                transform.RandomHorizontalFlip(1.0),
                transform.CenterCrop(size) if center_crop else transform.RandomCrop(size),
                transform.ToTensor(),
                transform.ImageNormalize([0.5], [0.5]),
            ]
        )
        
        self.set_order()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if len(self.order) == 0:
            self.set_order()
        style = self.order[-1][0]
        index = self.order[-1][1]
        self.order.pop()
        print(style, index)
        
        # style = self.styles[np.random.randint(0, len(self.styles))]
        # print(self.instance_images_path[style][index % self.num_instance_images[style]])
        # instance_image = Image.open(self.instance_images_path[style][index % self.num_instance_images[style]])
        
        instance_image = Image.open(self.instance_images_path[style][index])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        flag_flip = np.random.rand() < 0.5
        if flag_flip:
            example["instance_images"] = self.image_transforms_flip(instance_image)
        else:
            example["instance_images"] = self.image_transforms_no_flip(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            # trigger word + descriptions
            name = self.instance_images_path[style][index % self.num_instance_images[style]].name[:-4] # remove .png
            file = open(self.instance_data_root[style].parent / 'tag.json', 'r')
            tags = json.load(file)
            file.close()
            prompt = tags['item'][name] + ", " + name
            if flag_flip:
                prompt += ",horizontal_flip"
            
            def randomize_prompt(prompt):
                prompt = prompt.replace('  ',' ')
                a = prompt.split(',')
                random.shuffle(a)
                return ','.join(a)
            
            prompt = self.instance_prompt[style] + "," + tags['style'] + "," + randomize_prompt(prompt)
            prompt = prompt.replace('  ',' ')
            prompt = prompt.replace(',,',',')
            prompt = prompt.replace(', ',',')
            
            text_inputs = tokenize_prompt(
                self.tokenizer, prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            # print(text_inputs)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            print(prompt)

        if self.class_data_root:
            assert(0)

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = jt.stack(pixel_values)
    pixel_values = pixel_values.float()

    input_ids = jt.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    print(f"in prompt_tokenizer, prompt: {prompt}")
    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    jt.flags.use_cuda = 1
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    for name, param in text_encoder.named_parameters():
        assert param.requires_grad == False, name
    # print(text_encoder)
    
    encoder_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    text_encoder = get_peft_model(text_encoder, encoder_lora_config)
    # text_encoder.add_adapter("text_adapter",encoder_lora_config)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
   #if vae is not None:
   #    vae.requires_grad_(False)
   #text_encoder.requires_grad_(False)
   #unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = jt.float32

    for name, param in unet.named_parameters():
        assert param.requires_grad == False, name
    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.dropout,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    thresh = np.array([320,640,1024,1280])
    scale  = np.sqrt(thresh / args.rank)
    def get_larger_than_one(x):
        return max(x, 1)
    get_larger_than_one = np.vectorize(get_larger_than_one)
    scale = get_larger_than_one(scale)
    print(scale)
    # scale = np.array([1.0, 1.0, 1.0, 1.0])
    # lora_B: down
    # lora_A: up
    optimizer = AdamW(
        [
            {'params': [
                param for name, param in text_encoder.named_parameters() 
                    if param.requires_grad and ('lora_B' in name)
                ], 'lr': args.learning_rate
            },
            {'params': [
                param for name, param in text_encoder.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and np.max(param.shape) <= thresh[0]
                ], 'lr': args.learning_rate * scale[0]
            },
            {'params': [
                param for name, param in text_encoder.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[0] < np.max(param.shape) <= thresh[1]
                ], 'lr': args.learning_rate * scale[1]
            },
            {'params': [
                param for name, param in text_encoder.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[1] < np.max(param.shape) <= thresh[2]
                ], 'lr': args.learning_rate * scale[2]
            },
            {'params': [
                param for name, param in text_encoder.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[2] < np.max(param.shape)
                ], 'lr': args.learning_rate * scale[3]
            },
            
            
            {'params': [
                param for name, param in unet.named_parameters() 
                    if param.requires_grad and ('lora_B' in name)
                ], 'lr': args.learning_rate
            },
            {'params': [
                param for name, param in unet.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and np.max(param.shape) <= thresh[0]
                ], 'lr': args.learning_rate * scale[0]
            },
            {'params': [
                param for name, param in unet.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[0] < np.max(param.shape) <= thresh[1]
                ], 'lr': args.learning_rate * scale[1]
            },
            {'params': [
                param for name, param in unet.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[1] < np.max(param.shape) <= thresh[2]
                ], 'lr': args.learning_rate * scale[2]
            },
            {'params': [
                param for name, param in unet.named_parameters() 
                    if param.requires_grad and ('lora_A' in name) and thresh[2] < np.max(param.shape)
                ], 'lr': args.learning_rate * scale[3]
            },
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        pack=args.pack,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.num_process,
        num_training_steps=args.max_train_steps * args.num_process,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")

    # Train!
    total_batch_size = args.train_batch_size * args.num_process * args.gradient_accumulation_steps

    print(f"***** Running training: {model_name} *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Rank = {args.rank}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  Pack = {args.pack}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )
    
    # for name,param in unet.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
    # for name,param in text_encoder.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
    # exit()
    # print(text_encoder)
    # exit()
    
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") + f":{args.pack}"

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = jt.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = jt.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), 
            ).to(device=model_input.device)
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )
            # print(batch["input_ids"])
            # exit()

            if unet.config.in_channels == channels * 2:
                noisy_model_input = jt.cat([noisy_model_input, noisy_model_input], dim=1)

            if args.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = jt.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = nn.mse_loss(model_pred, target)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.detach().item()}
            with open(args.loss_dir+f"{formatted_time}_{model_name}.csv", "a") as f: 
                print(f"{global_step}, {loss.detach().item()}", file = f)
            #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % args.save_frequency == 0:
                # Save the lora layers
                unet_g = unet.to(jt.float32)
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_g))

                text_encoder_g = text_encoder.to(jt.float32)
                text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_g))
                LoraLoaderMixin.save_lora_weights(
                    save_directory=args.output_dir+f"_{model_name}_{global_step}",
                    unet_lora_layers=unet_lora_state_dict,
                    text_encoder_lora_layers=text_encoder_state_dict,
                    safe_serialization=False
                )
            if global_step >= args.max_train_steps:
                break


    # for name,param in unet.named_parameters():
    #     if param.requires_grad == True:
    #         print(name,param)
    # for name,param in text_encoder.named_parameters():
    #     if param.requires_grad == True:
    #         print(name,param)
    # Save the lora layers
    unet = unet.to(jt.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    text_encoder_g = text_encoder.to(jt.float32)
    text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_g))
    LoraLoaderMixin.save_lora_weights(
        save_directory=args.output_dir+f"_{model_name}",
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_state_dict,
        safe_serialization=False
    )
    # text_encoder.save_weights(save_directory=args.output_dir+"_add_encoder")

def set_seed(seed):
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
