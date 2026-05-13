import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
import wandb
import os
import re
import shutil

def compute_metrics(eval_preds):
    # Load the specific metrics for the MRPC task from the evaluate library
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    
    # Logits are raw mathematical outputs. We take the highest value to get our prediction (0 or 1).
    predictions = np.argmax(logits, axis=-1)
    
    # Compare predictions to the true labels
    return metric.compute(predictions=predictions, references=labels)



def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Paraphrase Detection on MRPC")
    
    # Data arguments
    parser.add_argument("--max_train_samples", type=int, default=-1, help="Number of training samples or -1 for all")
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="Number of validation samples or -1 for all")
    parser.add_argument("--max_predict_samples", type=int, default=-1, help="Number of prediction samples or -1 for all")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    
    # Execution flags
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_predict", action="store_true", help="Run prediction")
    parser.add_argument("--model_path", type=str, default=None, help="The model path to use when running prediction")

    args = parser.parse_args()

    # --- Step 1: Initialize Weights & Biases (wandb) if training ---
    if args.do_train:
        print("Starting training mode...")
        wandb.init(project="anlp-ex1-mrpc-final")

    
    print("Loading MRPC dataset and tokenizer...")
    
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataset = tokenized_datasets["train"]
    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    eval_dataset = tokenized_datasets["validation"]
    if args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        
    predict_dataset = tokenized_datasets["test"]
    if args.max_predict_samples != -1:
        predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}, Predict size: {len(predict_dataset)}")
    
    

    if args.do_train:
        print("Starting training mode...")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        
        # 1. Map your command-line arguments to Hugging Face's TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results", # Where to save the outputs
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy="epoch", # Evaluate at the end of every epoch
            report_to="wandb", 
            logging_steps=1,
            save_strategy="no"
        )

        # 2. Instantiate the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 3. START TRAINING!
        print("Commencing training loop...")
        trainer.train()
        
        val_acc = trainer.evaluate()['eval_accuracy']
        path = "/notebooks/"
        pattern = r"my_final_model_acc([\d.]+)"
        
        found_existing = False
        save_new_model = False

        # Check existing folders
        for folder_name in os.listdir(path):
            match = re.search(pattern, folder_name)
            if match:
                found_existing = True
                past_accuracy = float(match.group(1))
                
                if val_acc > past_accuracy:
                    print(f"Improved! Past acc: {past_accuracy}, New acc: {val_acc}")
                    last_model_path = os.path.join(path, folder_name)
                    shutil.rmtree(last_model_path)
                    save_new_model = True
                else:
                    print(f"No improvement. New acc {val_acc} is not better than {past_accuracy}")
                    save_new_model = False
                
                break # Found the model folder, no need to keep looking

        # Logic for saving
        # Save if it's the first model OR if it's an improvement
        if not found_existing or save_new_model:
            save_path = f"./my_final_model_acc{val_acc}"
            print(f"Saving model to {save_path}")
            trainer.save_model(save_path)        
        

    if args.do_predict:
        print("Starting prediction mode...")
        if not args.model_path:
            raise ValueError("--model_path must be specified when using --do_predict")
        
        # 1. Load the specifically trained model
        print(f"Loading model from {args.model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
        
        # 2. Set to evaluation mode (Safety Check!)
        model.eval()
        
        # 3. Create a bare-bones prediction engine
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
        )
        
        # 4. Generate the predictions
        print("Running prediction on the test set...")
        prediction_output = trainer.predict(predict_dataset)
        
        # 5. Extract logits and convert to 0s and 1s
        logits = prediction_output.predictions
        predictions = np.argmax(logits, axis=-1)
        
        output_file = "predictions.txt"
        print(f"Writing formatted predictions to {output_file}...")
        print("DEBUG: I am officially running the NEW formatting loop!")
        with open(output_file, "w") as writer:
            for i, pred in enumerate(predictions):
                # Grab the original sentences from the dataset using the index 'i'
                s1 = predict_dataset[i]["sentence1"]
                s2 = predict_dataset[i]["sentence2"]
                
                # Write them in the exact format requested: <s1>###<s2>###<label>
                writer.write(f"{s1}###{s2}###{pred}\n")
                
        print("Done! Formatted predictions saved.")        
        
if __name__ == "__main__":
    main()