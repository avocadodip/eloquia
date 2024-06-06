import torch
from data_utils import SoundDS, load_data
from model_utils import get_model
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments

def main():
    data_path = "data/ground_truth.csv"
    device = torch.device("cpu")
    
    df = load_data()
    model, feature_extractor = get_model(device)

    myds = SoundDS(df, "", device)
    num_items = len(myds)
    num_train = round(num_items * 0.7)
    num_val = round(num_items * 0.15)
    num_test = num_items - num_train - num_val
    train_ds, val_ds, test_ds = random_split(myds, [num_train, num_val, num_test])

    args = TrainingArguments(
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        output_dir="./"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor
    )

    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()
