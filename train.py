import torch
from data_utils import SoundDS, load_data
from model_utils import get_model
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
from transformers import get_scheduler
import numpy as np

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def stratified_split(dataset, test_size=0.2, val_size=0.1):
    # Extract class labels from the dataset
    labels = [dataset.df.loc[i, 'classID'] for i in range(len(dataset))]
    
    # Split the dataset indices into training and temporary (test + validation) sets
    train_idx, temp_idx, _, _ = train_test_split(
        range(len(labels)),
        labels,
        stratify=labels,
        test_size=test_size + val_size,
        random_state=42  # for reproducibility
    )
    
    # Now split the temporary set into test and validation sets
    val_size_adjusted = val_size / (test_size + val_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        [labels[i] for i in temp_idx],
        stratify=[labels[i] for i in temp_idx],
        test_size=val_size_adjusted,
        random_state=42  # for reproducibility
    )
    
    # Create subset objects
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    return train_ds, val_ds, test_ds

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """ Set up the custom optimizer and learning rate scheduler. """
        # Custom AdamW optimizer with specific hyperparameters
        optimizer = AdamW(self.model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
        
        # Linear learning rate scheduler with warmup
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),  # Warmup
            num_training_steps=num_training_steps
        )
        
        self.lr_scheduler = scheduler
        self.optimizer = optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """ Custom loss function computation. """
        # Assuming model outputs are in a dictionary `outputs` with key 'logits'
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get("labels")

        # Using CrossEntropyLoss as an example
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average='macro')}  # or average='binary' based on your problem

def main():
    data_path = "data/ground_truth.csv"
    device = torch.device("cuda")
    
    df = load_data()
    model, feature_extractor = get_model(device)

    myds = SoundDS(df, "", device)
    
    # num_items = len(myds)
    # num_train = round(num_items * 0.7)
    # num_val = round(num_items * 0.15)
    # num_test = num_items - num_train - num_val
    # train_ds, val_ds, test_ds = random_split(myds, [num_train, num_val, num_test])

    train_ds, val_ds, test_ds = stratified_split(myds, test_size=0.15, val_size=0.15)

    args = TrainingArguments(
        output_dir="./results",                # output directory
        eval_strategy="epoch",                 # evaluate at the end of each epoch
        save_strategy="epoch",                 # save model at the end of each epoch
        learning_rate=6.25e-6,                 # starting learning rate
        per_device_train_batch_size=16,         # batch size for training
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=16,         # batch size for evaluation
        gradient_checkpointing=True,
        fp16=True,
        num_train_epochs=5,                    # number of training epochs
        warmup_ratio=0.1,                      # warmup steps as a ratio of total steps
        logging_dir='./runs',                  # directory for storing logs
        logging_steps=1,                      # log every 10 steps
        load_best_model_at_end=True,           # load the best model at the end of training
        metric_for_best_model="f1",            # use f1 to find the best model
        report_to="tensorboard",             # use tensorboard
        label_names=["labels"],
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()
