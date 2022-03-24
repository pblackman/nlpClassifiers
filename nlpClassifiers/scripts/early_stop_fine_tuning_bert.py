import matplotlib.pyplot as plt
import torch.nn as nn
from os.path import join
import torch
from nlpClassifiers.data.dataset  import NLPDataset
#from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.nn import LayerNorm as BertLayerNorm
import numpy as np
import time
import logging
import datetime
import random
import pandas as pd
import argparse
import pickle as pk
import itertools
import os
import shutil
from pathlib import Path
import copy
import wandb
import re
from nlpClassifiers import settings
from scipy.special import expit
from sklearn.metrics import classification_report

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

parser = argparse.ArgumentParser(description="Arguments for fine-tuning.")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--dataset", type=str)
parser.add_argument("--save-stats", type=int, default=0)
parser.add_argument("--save-name", type=str, default=None)
parser.add_argument("--save-model", type=int, default=0)
parser.add_argument("--bert-size", type=str, default="base") # "base" or "large"
parser.add_argument("--layerwise-lr", type=float, default=0.9)
parser.add_argument("--wandb", type=int, default=0)
parser.add_argument("--reset-layers", type=int, default=0)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--early-stop-patience", type=int, default=4)
parser.add_argument("--bert-path", type=str, default="")
parser.add_argument("--sentence-max-len", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)


args = parser.parse_args()
gpu = args.gpu
dataset = args.dataset
save_stats = args.save_stats
save_name = args.save_name
save_model = args.save_model
bert_size = args.bert_size
layerwise_lr = args.layerwise_lr
log_wandb = args.wandb
reset_layers = args.reset_layers
sentence_max_len = args.sentence_max_len
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
patience = args.early_stop_patience
bert_path = args.bert_path
seed = args.seed

save_data = []

BASE_PATH_TO_MODELS = {"virtual-operator": settings.PATH_TO_VIRTUAL_OPERATOR_MODELS, "agent-benchmark": settings.PATH_TO_AGENT_BENCHMARK_MODELS, "mercado-livre-pt": settings.PATH_TO_ML_PT_MODELS}
FULL_PATH_TO_MODELS = join(BASE_PATH_TO_MODELS[dataset], "bert-base-portuguese-tapt-classifier")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_lr_by_layer(name: str, base_lr: float = 2e-5, decay: int = 0.9, bert_size = "base"):
    match = re.search("^.*\\.(\d+)\\..*$", name)
    last_layer_idx = 23 if bert_size == "large" else 11
    if match is None:
        if "embeddings" in name:
            # last_layer_idx + 2 is because there is one more 
            # pooler and classifier layer on top of the 11
            # stack of encoders.
            return base_lr * (decay ** (last_layer_idx + 2))
        elif "pooler" in name:
            return base_lr * (decay ** 1)
        elif "classifier" in name:
            return base_lr
    else:
        layer = int(match.group(1))
        return base_lr * (decay ** (last_layer_idx + 2 - layer))


def predict(
    model_path: Path,
    dataset: str,
    batch_size: int,
    labels_dict,
    device: torch.device
):
       
    print(f"====Loading dataset for testing")
    test_corpus = NLPDataset(dataset, "test", sentence_max_len, bert_path, labels_dict)
    test_dataloader = DataLoader(
        test_corpus,
        batch_size=batch_size,
        sampler = RandomSampler(test_corpus),
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )

    print(f"====Loading model for testing")
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels = test_corpus.num_labels,
        output_attentions = False,
        output_hidden_states = True,
    )
    model.to(device)
    model.eval()
  #  cm = ConfusionMatrix([0,1])
    pred_labels = []
    test_labels = []
    logits_list = []

    def _list_from_tensor(tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        return list(tensor.cpu().detach().numpy())

    print("====Testing model...")
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_segment_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            loss, logits, *_ =  model(b_input_ids, b_input_mask, token_type_ids=None, labels=b_labels)

            preds = np.argmax(logits.cpu(), axis=1) # Convert one-hot to index
            b_labels = b_labels.int()
            pred_labels.extend(_list_from_tensor(preds))
            test_labels.extend(_list_from_tensor(b_labels))
        logits_list.extend(_list_from_tensor(logits))

    class_results_dict = classification_report(test_labels, pred_labels, labels=list(labels_dict.values()), target_names=np.array(list(labels_dict.keys())), digits=3, output_dict=True)
    class_results_text = classification_report(test_labels, pred_labels, labels=list(labels_dict.values()), target_names=np.array(list(labels_dict.keys())), digits=3, output_dict=False)
    logging.info(class_results_text)
    logits_list = expit(logits_list)

    if log_wandb:
        wandb.log(
            {
                "accuracy": class_results_dict["accuracy"],
                "macro avg precision": class_results_dict["macro avg"]["precision"],
                "macro avg recall": class_results_dict["macro avg"]["recall"],
                "macro avg f1-score": class_results_dict["macro avg"]["f1-score"],
                "macro avg support": class_results_dict["macro avg"]["support"],
                "weighted avg precision": class_results_dict["weighted avg"]["precision"],
                "weighted avg recall": class_results_dict["weighted avg"]["recall"],
                "weighted avg f1-score": class_results_dict["weighted avg"]["f1-score"],
                "weighted avg support": class_results_dict["weighted avg"]["support"]                
            }
        )

    del model
    torch.cuda.empty_cache()

def get_accuracy_from_logits(logits, labels):
    acc = (labels.cpu() == logits.cpu().argmax(-1)).float().detach().numpy()
    return float(100 * acc.sum() / len(acc))

# device = torch.device(f"cuda:{gpu}")
device = torch.device(f"cuda:{gpu}")

best_epoch = -1
last_saved_model = ""

print(f"Parameters: {vars(args)}")

best_val_acc = float("-inf")
best_model_wts = None
# for comb in :

best_curr_val = 0

if log_wandb:
    wandb_conf = vars(args)
    wandb.init(project="huggingface", config=wandb_conf, reinit=True)

def _init_fn(worker_id):
    np.random.seed(seed)

train_corpus = NLPDataset(dataset, "train", sentence_max_len, bert_path)
labels_dict = train_corpus.labels_dict
val_corpus = NLPDataset(dataset, "val", sentence_max_len, bert_path, labels_dict)

# HYPERPARAMS
n_epochs_no_improvement = 0
gradient_accumulation_steps = 1

set_seed(seed)
print(f"Loading dataset ...")
train_dataloader = DataLoader(
            train_corpus,
            sampler = RandomSampler(train_corpus),
            batch_size = batch_size,
            pin_memory=True,
            worker_init_fn=_init_fn,
            num_workers=0
)

validation_dataloader = DataLoader(
            val_corpus, 
            sampler = RandomSampler(val_corpus),
            batch_size = batch_size,
            pin_memory=True,
            worker_init_fn=_init_fn,
            num_workers=0
)

print(f"Loading bert_model...")

print(f"Using BertForSequenceClassification")
model = BertForSequenceClassification.from_pretrained(
    bert_path,
    num_labels = train_corpus.num_labels,
    output_attentions = False,
    output_hidden_states = False
)

set_seed(seed)
nn.init.normal_(model.classifier.weight.data, 0, 0.02)
nn.init.zeros_(model.classifier.bias.data)

# We reset the last layer dense and LayerNorm parameters
def reset_last_layers(n_layers: int):
    last_layer_idx = 23 if bert_size == "large" else 11
    first = last_layer_idx - n_layers + 1
    last = last_layer_idx + 1
    for i in range(first, last):
        print(f"====>Reseting layer {i}!")
        for name, module in model.bert.encoder.layer[i].named_modules():
            if any(x in name for x in ["dense", "query", "key", "value"]):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

        model.bert.encoder.layer[i].output.LayerNorm = BertLayerNorm(model.bert.encoder.layer[i].output.dense.out_features, eps=1e-12)

reset_last_layers(n_layers=reset_layers)

model = model.to(device)
if log_wandb:
    wandb.watch(model)

no_decay = ["bias", "LayerNorm.weight"]  # no weight decay for these params

if layerwise_lr != 0:
    print(f"====>Using layerwise learning rate with decay={layerwise_lr}")
    optimizer_grouped_parameters = []
    for name, p in model.named_parameters():
        if not any(nd in name for nd in no_decay) and p.requires_grad:
            d = {"params": p, "weight_decay": 0.01, "lr": get_lr_by_layer(name, lr, layerwise_lr,  "base")}
        elif any(nd in name for nd in no_decay) and p.requires_grad:
            d = {"params": p, "weight_decay": 0.00, "lr": get_lr_by_layer(name, lr, layerwise_lr, "base")}
        
        optimizer_grouped_parameters.append(d)

else:
    print(f"====>Using fixed learning rate={lr}")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": 0.0
        },
    ]

#optimizer = SGD(model.parameters(), lr)

optimizer = AdamW(
    optimizer_grouped_parameters, 
    lr=lr,
    betas=(0.9, 0.999),
    eps=0.000001, 
)

t_total = (len(train_dataloader) // gradient_accumulation_steps) * epochs
print(f"====>TOTAL NUMBER OF STEPS: {t_total}")
warmup_steps = int(t_total * 0.1) # 10% of total steps during fine-tuning
print(f"====>WARMUP STEPS: {warmup_steps}")

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = warmup_steps,
    num_training_steps = t_total # We used total_steps before.... Why?
)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


training_stats = []
total_t0 = time.time()
# mean_val_acc_over_epochs = 0
global_step = 0
torch.cuda.empty_cache()
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: segment_ids
        #   [2]: attention masks    
        #   [3]: labels 
        # print(batch[2])
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.

        loss, logits, *_ = model(b_input_ids, b_input_mask, token_type_ids=None, labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        total_train_loss += loss.item()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if log_wandb and global_step % 100 == 0:
            layer_grads = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grads = param.grad.detach().cpu()
                grads = grads.view(-1)
                match = re.search("^.*\\.(\d+)\\..*$", name)
                if match is not None:
                    layer = f"Layer {match.group(1)}"
                    if layer not in layer_grads: layer_grads[layer] = []
                    layer_grads[layer].append(grads)
                else:
                    if "classifier" in name:
                        layer = "classifier"
                        if layer not in layer_grads: layer_grads[layer] = []
                        layer_grads[layer].append(grads)
                    elif "pooler" in name:
                        layer = "pooler"
                        if layer not in layer_grads: layer_grads[layer] = []
                        layer_grads[layer].append(grads)
            
            layer_norms = {}
            for k, grads in layer_grads.items():
                layer_norms[k] = torch.cat(grads).norm(p=2)
            wandb.log(layer_norms)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        # if global_step % 40 == 0:
            # print(f"====>Current learning rate: {optimizer.param_groups[0]['lr']}")
            # Another way to get the current learning rate
            # print(f"====>Current learning rate: {scheduler.get_lr()[0]}")

        global_step += 1

    # Calculate the average loss over all of the batches.
    # print(f"====>SIZE OF TRAIN DATALOADER={len(train_corpus)}")
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0.0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: segment ids
        #   [2]: attention masks
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            loss, logits, *_ = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss
        batch_acc = get_accuracy_from_logits(logits, b_labels)
        print("Batch accuracy: {0:.2f}".format(batch_acc))
        total_eval_accuracy += batch_acc

        # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    # print(f"====> Avg_val_accuracy type={}")
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    if log_wandb:
        wandb.log({"epoch": epoch_i, "loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": avg_val_accuracy})
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    # Early Stopping

    if avg_val_accuracy > best_val_acc:
        print(f"New best model, saving it!")
        if avg_val_accuracy > best_curr_val:
            best_curr_val = avg_val_accuracy
        if last_saved_model:
            shutil.rmtree(last_saved_model)
        model_path = Path(
            FULL_PATH_TO_MODELS, 
            f"base-dataset-{dataset}-{save_name}"
        )
        last_saved_model = model_path
        model_path.mkdir(parents=True, exist_ok=True)
        best_val_acc = avg_val_accuracy
        model.save_pretrained(model_path)
        best_epoch = epoch_i
        n_epochs_no_improvement = 0
    elif avg_val_accuracy > best_curr_val:
        best_curr_val = avg_val_accuracy
        n_epochs_no_improvement = 0
    else:
        n_epochs_no_improvement += 1
        print(f"The model does not improve for {n_epochs_no_improvement} epochs!")
    
    if n_epochs_no_improvement > patience:
        print(f"====>Stopping training, the model did not improve for {n_epochs_no_improvement}\n====>Best epoch: {best_epoch}.")
        break

    # mean_val_acc_over_epochs += avg_val_accuracy.item()

# mean_val_acc_over_epochs /= epochs
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

if log_wandb:
    wandb.log({"best_epoch": best_epoch})

print(f"Training stats: {training_stats}")

# print(f"Mean Valid. Acc. over epochs: {mean_val_acc_over_epochs}")

save_data.append({"training_stats":training_stats})

model.to("cpu")

del train_dataloader
del validation_dataloader
del train_corpus
del val_corpus
del model
torch.cuda.empty_cache()


predict(last_saved_model, dataset, batch_size, labels_dict, device)

if save_stats:
    if save_name:
        save_path = os.path.join(FULL_PATH_TO_MODELS, f"{bert_size}-dataset-{dataset}-{save_name}.pk")
    else:
        save_path = os.path.join(FULL_PATH_TO_MODELS, f"{bert_size}-dataset-{dataset}.pk")
    with open(save_path, "wb") as f:
        pk.dump(save_data, f)