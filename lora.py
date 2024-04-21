from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import wandb
from typing import Dict

from callback import EarlyStoppingCallback

# from torchmetrics.text.bleu import BLEUScore


class Lora_FineTuner:
    """
    QLORA Fine-tuning using Unsloth, Huggingface Transformers.
    """
    
    def __init__(self, dataset, model_name, load_in_4bit, max_seq_length, 
            model_args: Dict, training_args: Dict) -> None:
        self.model_name = model_name
        self.dataset = load_from_disk(dataset)
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit,
        self.model_args = model_args
        self.training_args = training_args
        # if wandb_track: self._setup_wandb()
        self._setup_model_and_tokenizer()
    
    def _setup_wandb(self):
        pass


    def _setup_model_and_tokenizer(self):
        model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name = self.model_name, 
        max_seq_length = self.max_seq_length,
        dtype = None,
        load_in_4bit = self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            model,
            **self.model_args,
            random_state=42,
        )
    
    def _setup_trainer(self, n_rows):
        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset['train'],
            eval_dataset = self.dataset['eval'],
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.05)],
            # compute_metrics=self._compute_F1,
            args = TrainingArguments(
                **self.training_args
            ),
        )
    
    def train(self, resume=False, n_rows = None):
        self._setup_trainer(n_rows=n_rows)
        trainer_stats = self.trainer.train(resume_from_checkpoint = resume)
        print(trainer_stats)