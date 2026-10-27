from typing import List
import argparse
import copy
import functools
import json
import os
import sys
from multiprocessing import Pool

import torch
from arclib import arc
from torchtune.config._parse import TuneRecipeArgumentParser
from torchtune.config._utils import _merge_yaml_and_cli_args
from torchtune.models.llama3 import llama3_tokenizer

import arclib.messagers
from arclib.arc import read_tasks_from_single_file
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from ttt.preprocess import get_augmenters, process_task


sys.path.append("third_party/torchtune/recipes/")
import lora_finetune_single_device


def save_adapter_config(
    path: str,
    base_model_path: str,
    lora_rank: int = 64,
    peft_type: str = "LORA",
    lora_alpha: float = 16.0,
    lora_attn_modules: List[str] = ["q_proj", "v_proj"],
    lora_to_mlp: bool = True,
    lora_to_output: bool = False,
):
    # This config is used by VLLM
    # Target modules is not the same as we use in training, but rather inclusive of all
    # because I was getting weird bugs in VLLM part
    target_modules = []
    if lora_to_mlp:
        target_modules += ["gate_proj", "down_proj", "up_proj"]
    if lora_to_output:
        target_modules += ["lm_head"]
    if lora_attn_modules:
        target_modules += lora_attn_modules

    config = {
        "base_model_name_or_path": base_model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": peft_type,
        "r": lora_rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    with open(path, "w") as f:
        json.dump(config, f)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--data_file",
    type=str,
    default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
    help="Data file path to evaluate",
)
parser.add_argument(
    "--num_tasks", type=int, default=None, help="Number of tasks to process for limited evaluation."
)
parser.add_argument(
    "--base_checkpoint",
    type=str,
    default="checkpoints/pretrained/all_in_fix_final_checkpoints/",
    help="path to the pretrained checkpoint",
)
parser.add_argument(
    "--lora_checkpoints_folder",
    type=str,
    default="checkpoints/ttt/all_in_fix_final_lora_clean/",
    help="LoRA checkpoints folder, if none then base model is used",
)
parser.add_argument(
    "--quantization", type=str, default=None, help="Quantization type bitsandbytes or none"
)
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--cpus", type=int, default=64, help="Number of cpus")
parser.add_argument(
    "--lora_config",
    type=str,
    default="configs/ttt/8B_lora_single_device.yaml",
    help="LoRA config file",
)
parser.add_argument(
    "--experiment_folder", type=str, default="experiments/ttt/new/", help="submission folder"
)
parser.add_argument(
    "--formatter",
    type=str,
    default="arclib.messagers.GPTTextMessageRepresenterV2",
    help="formatter for the task, better to be same with the one used for training",
)
parser.add_argument("--unmask_outputs", type=bool, default=True, help="Unmask outputs setting")
parser.add_argument("--train_on_input", type=bool, default=False, help="Train on input setting")
parser.add_argument("--permute_n", type=int, default=1, help="Permute n")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument(
    "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
)
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--compile", type=bool, default=True, help="Compile setting")
parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
parser.add_argument(
    "--lora_attn_modules", type=str, nargs="+", default=["q_proj", "v_proj"], help="LoRA parameters"
)
parser.add_argument("--lora_to_mlp", type=bool, default=True, help="Apply LoRA to MLP")
parser.add_argument("--lora_to_output", type=bool, default=False, help="Apply LoRA to output")
parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument(
    "--base_checkpoint_dir",
    type=str,
    default="checkpoints/pretrained/multi_format_model/",
    help="Checkpoint directory",
)
parser.add_argument(
    "--new_format", action="store_true", help="Whether to use the new format or not"
)

parser.add_argument(
    "--barc_format", action="store_true", help="Whether to use the barc format or not"
)

parser.add_argument(
    "--no_transform", action="store_true", help="Whether to use the new format or not"
)



args = parser.parse_args()

os.makedirs(args.experiment_folder, exist_ok=True)

arc_test_tasks = read_tasks_from_single_file(args.data_file, test=True)

# # reverse
# arc_test_tasks = arc_test_tasks[::-1][:200]

arc_test_tasks = [task for task in arc_test_tasks if "-0" in task.name]
if args.num_tasks is not None:
    arc_test_tasks = arc_test_tasks[: args.num_tasks]
arc_test_ids = [task.name.replace("-0", "") for task in arc_test_tasks]

print("Number of train tasks: ", len(arc_test_tasks))

if args.new_format:
    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    formatter = GPTTextMessageRepresenterV2(task_representer=standard_formatter)
elif args.barc_format:
    formatter = arclib.messagers.GPTTextMessageRepresenterForBarc(
        prompt = (
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n"
            "You are a world-class puzzle solver with exceptional pattern recognition skills. "
            "Your task is to analyze puzzles, spot patterns, and provide direct solutions."
        ),
        task_representer=arclib.representers.TextTaskRepresenter(
            example_representer=arclib.representers.TextExampleRepresenter(
            grid_representer=arclib.representers.WordGridRepresenter(),
            input_header="Input:\n",
            output_header="\nOutput:\n",
            io_sep="\n"
        )))

else:
    formatter = arclib.messagers.GPTTextMessageRepresenterV2()

# Load config
conf = _merge_yaml_and_cli_args(
    *TuneRecipeArgumentParser(
        description="LORA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_known_args(["--config={}".format(args.lora_config)])
)

# Update conf with argparse settings
conf.dataset.unmask_outputs = args.unmask_outputs
conf.dataset.train_on_input = args.train_on_input
conf.epochs = args.epochs
conf.batch_size = args.batch_size
conf.gradient_accumulation_steps = args.gradient_accumulation_steps
conf.optimizer.lr = args.learning_rate
conf.compile = False  # we will do it ourselves
conf.model.lora_rank = args.lora_rank
conf.model.lora_alpha = args.lora_alpha
conf.model.lora_attn_modules = args.lora_attn_modules
conf.model.apply_lora_to_mlp = args.lora_to_mlp
conf.model.lora_dropout = args.lora_dropout
conf.checkpointer.checkpoint_dir = args.base_checkpoint_dir
conf.seed = args.seed

if "llama3_2" not in conf.model._component_:
    conf.model.apply_lora_to_output = args.lora_to_output
else:
    print("Ignoring lora_to_output for llama3_2")
    # conf.model.lora_to_output = False

# print conf
print(conf)

tokenizer = llama3_tokenizer(conf.tokenizer.path)

if args.no_transform:
    augmenters_to_apply = []
else:
    augmenters_to_apply = get_augmenters(
        include_basic=True, include_size=True, include_chain=True, include_repeat=True
    )


def train_with_a_test_data(
    task_id: str,
    recipe,
    conf,
    experiment_folder: str,
    model=None,
    adapter=None,
):
    recipe.cleanup()
    lconf = copy.deepcopy(conf)
    lconf.dataset.source = f"{experiment_folder}/{task_id}"
    lconf.output_dir = f"{experiment_folder}/{task_id}"
    lconf.checkpointer.output_dir = f"{experiment_folder}/{task_id}"

    with open(
        f"{experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl"
    ) as f:
        train_data_size = sum(1 for _ in f)

    print("Train data size: ", train_data_size)

    total_steps = (train_data_size * lconf.epochs) // (
        lconf.batch_size * lconf.gradient_accumulation_steps
    )
    lconf.lr_scheduler.num_warmup_steps = total_steps // 10

    print(f"====CONFIG FOR {task_id}====")
    for k, v in lconf.items():
        print(f"{k}: {v}")
    print("=============================")

    recipe._output_dir = lconf.output_dir
    recipe.total_epochs = lconf.epochs
    recipe._gradient_accumulation_steps = lconf.gradient_accumulation_steps
    recipe.global_step = 0
    recipe.epochs_run = 0

    recipe.seed = lora_finetune_single_device.training.set_seed(lconf.seed)
    recipe.setup(cfg=lconf, model=model, adapter=adapter)

    for layer in recipe._model.layers:
        layer.attn.kv_cache = None
        recipe._model.causal_mask = None

    recipe.train()
    recipe._optimizer = None

    return recipe._model


processor = functools.partial(
    process_task,
    augmenters=augmenters_to_apply,
    formatter=formatter,
    tokenizer=tokenizer,
    permute_n=args.permute_n,
    Nmax=250,
    seed=args.seed,
)

with Pool(args.cpus) as p:
    data = p.map(processor, arc_test_tasks)

# data = [processor(task) for task in arc_test_tasks]

assert len(data) == len(arc_test_tasks)

for task, task_train_data in zip(arc_test_tasks, data):
    task_id = task.name.replace("-0", "")

    os.makedirs(f"{args.experiment_folder}/{task_id}", exist_ok=True)

    with open(
        f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
        "w",
    ) as f:
        for td in task_train_data:
            print(json.dumps(td), file=f)
    # we need a placeholder test file for torchtune
    with open(
        f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
        "r",
    ) as src, open(
        f"{args.experiment_folder}/{task_id}/td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
        "w",
    ) as dst:
        first_line = src.readline()
        dst.write(first_line)


# initialize model
recipe = lora_finetune_single_device.LoRAFinetuneRecipeSingleDevice(conf)
recipe.setup(cfg=conf)
model = recipe._model
device = recipe._device
# compile model
# if args.compile:
#     model.compile(backend="inductor")

adapter = copy.deepcopy(lora_finetune_single_device.get_adapter_params(model))

for task in arc_test_tasks:
    task_id = task.name.replace("-0", "")
    # task_id = "all"
    print(f"Trying task {task_id}")
    try:
        adapter_path = f"{args.experiment_folder}/{task_id}/adapter_model.bin"
        # if exist continue
        if os.path.exists(adapter_path):
            print(f"Adapter for {task_id} already exists, skipping")
            continue
        test_file = f"{args.experiment_folder}/{task_id}/td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl"
        train_with_a_test_data(
            task_id,
            recipe,
            conf,
            experiment_folder=args.experiment_folder,
            model=model,
            adapter=adapter,
        )
        # save the adapter
        final_adapter = lora_finetune_single_device.get_adapter_params(model)
        # save
        replacements = {
            "_orig_mod.": "",
            "layers.": "base_model.model.model.layers.",
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
            ".attn": ".self_attn",
            "_checkpoint_wrapped_module.": "",
            "lora_b": "lora_B",
            "lora_a": "lora_A",
            "output.lora_": "base_model.model.lm_head.lora_",
        }

        saved_dict = {}
        for name, param in final_adapter.items():
            for old, new in replacements.items():
                name = name.replace(old, new)
            saved_dict[name] = param

        torch.save(saved_dict, adapter_path)
        saved_dict = None
        adapter_config_path = f"{args.experiment_folder}/{task_id}/adapter_config.json"
        save_adapter_config(
            adapter_config_path,
            args.base_checkpoint_dir,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_attn_modules=args.lora_attn_modules,
            lora_to_mlp=args.lora_to_mlp,
            lora_to_output=args.lora_to_output,
        )
    except Exception as e:
        print(e)
        print("Error training for ", task_id)
        continue
