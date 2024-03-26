from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import wandb

class Lora_FineTuner:
    """
    QLORA Fine-tuning using Unsloth, Huggingface Transformers.
    """
    
    def __init__(self, dataset, model_name: str, max_seq_length: int, 
                 load_in_4bit: bool = True, wandb_track: bool = True,
                 lora_rank=16, rs_lora=False) -> None:
        self.model_name = model_name
        self.dataset = load_from_disk(dataset)
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit,
        self.rank = lora_rank
        self.rs_lora = rs_lora
        if wandb_track: self._setup_wandb()
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
            r = self.rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = self.rs_lora,
            loftq_config = None,
        )
    
    def _setup_trainer(self, n_epochs, device_batch_size, n_rows, save_steps,
                      eval_steps):
        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset['train'].select(range(n_rows)) if n_rows is not None and n_rows > 0 else self.dataset['train'],
            eval_dataset = self.dataset['eval'].select(range(n_rows)) if n_rows is not None and n_rows > 0 else self.dataset['eval'],
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = device_batch_size,
                gradient_accumulation_steps = 4,
                save_steps=save_steps,
                warmup_steps = 5,
                learning_rate = 2e-4,
                num_train_epochs=n_epochs,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 50,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                eval_steps=eval_steps,
                evaluation_strategy='steps',
                do_eval = True,
                lr_scheduler_type = "linear",
                seed = 42,
                # report_to = "wandb",
                output_dir = "outputs",
            ),
        )
    
    def train(self, n_epochs, save_steps, eval_steps, resume, device_batch_size=8, n_rows = None):
        self._setup_trainer(n_epochs, device_batch_size=device_batch_size, 
                            n_rows=n_rows, save_steps=save_steps, eval_steps=eval_steps)
        trainer_stats = self.trainer.train(resume_from_checkpoint = resume)
