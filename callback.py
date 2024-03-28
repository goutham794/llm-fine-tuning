from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int, early_stopping_threshold: float):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_score = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_metric = kwargs.get("metrics", {}).get("eval_loss", None)  # Replace eval_loss with your evaluation metric
        if eval_metric is not None:
            if self.best_score is None or eval_metric < self.best_score + self.early_stopping_threshold:
                self.best_score = eval_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    control.should_training_stop = True
