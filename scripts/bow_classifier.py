# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# https://colab.research.google.com/drive/1ZQvuAVwA3IjybezQOXnrXMGAnMyZRuPU#scrollTo=E_t4cM6KLc98
import matplotlib.pyplot as plt
import torch.nn as nn
from os.path import join
import torch
from nlpClassifiers.data.dataset  import NLPDataset, Vocabulary
from nlpClassifiers.models.models import BOWClassifier
from torch.optim import SGD
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
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
parser.add_argument("--wandb", type=int, default=0)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--early-stop-patience", type=int, default=4)
parser.add_argument("--sentence-max-len", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--stopwords-lang", type=str, default=None)
parser.add_argument("--max-vocab-size", type=int, default=0)


args = parser.parse_args()
gpu = args.gpu
dataset = args.dataset
save_stats = args.save_stats
save_name = args.save_name
save_model = args.save_model
log_wandb = args.wandb
sentence_max_len = args.sentence_max_len
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
patience = args.early_stop_patience
seed = args.seed
debug = args.debug
stopwords_lang = args.stopwords_lang
max_vocab_size = args.max_vocab_size

save_data = []

BASE_PATH_TO_MODELS = {"virtual-operator": settings.PATH_TO_VIRTUAL_OPERATOR_MODELS, "agent-benchmark": settings.PATH_TO_AGENT_BENCHMARK_MODELS, "mercado-livre-pt": settings.PATH_TO_ML_PT_MODELS}
FULL_PATH_TO_MODELS = join(BASE_PATH_TO_MODELS[dataset], "bow-classifier")

def predict(
    model_path: Path,
    dataset: str,
    batch_size: int,
    labels_dict,
    device: torch.device
):

    print(f"====Loading dataset for testing")
    test_corpus = NLPDataset(dataset, "test", sentence_max_len, labels_dict = labels_dict, vocab= voc)

    test_dataloader = DataLoader(
        test_corpus,
        batch_size=batch_size,
        sampler = RandomSampler(test_corpus),
        pin_memory=True,
        num_workers=0,
        drop_last=False
    )

    print(f"====Loading model for testing")
    model = torch.load(join(model_path, "best-model.pth"))
    model.to(device)
    model.eval()
    pred_labels = []
    test_labels = []
    logits_list = []

    def _list_from_tensor(tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        return list(tensor.cpu().detach().numpy())

    print("====Testing model...")
    for batch in test_dataloader:
        bow_vector = batch[0].to(device)
        b_labels = batch[1].to(device)
        with torch.no_grad():
            loss, logits= model(bow_vector, b_labels)
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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def _init_fn(worker_id):
    np.random.seed(seed)

def get_accuracy_from_logits(logits, labels):
    acc = (labels.cpu() == logits.cpu().argmax(-1)).float().detach().numpy()
    return float(100 * acc.sum() / len(acc))


def read_data(dataset, subset):
    BASE_PATH_TO_DATASET = {"virtual-operator": settings.PATH_TO_VIRTUAL_OPERATOR_DATA, "agent-benchmark": settings.PATH_TO_AGENT_BENCHMARK_DATA, "mercado-livre-pt": settings.PATH_TO_ML_PT_DATA}
    BASE_PATH_TO_DATASET = {"train": join(BASE_PATH_TO_DATASET[dataset], "train.csv"), "val": join(BASE_PATH_TO_DATASET[dataset], "val.csv"), "test": join(BASE_PATH_TO_DATASET[dataset], "test.csv")}
    FULL_PATH_TO_DATASET = BASE_PATH_TO_DATASET[subset]
    
    if dataset == "mercado-livre-pt":
        sep=","
    else:
        sep=";"
    data = pd.read_csv(FULL_PATH_TO_DATASET, sep=sep, names =['utterance','label'], header=None, dtype={'utterance':str, 'label': str} )
    return data

set_seed(seed)

device = torch.device(f"cuda:{gpu}")

best_epoch = -1
last_saved_model = ""

print(f"Parameters: {vars(args)}")

best_val_acc = float("-inf")
best_model_wts = None
best_curr_val = 0

if log_wandb:
    wandb_conf = vars(args)
    wandb.init(project="huggingface", config=wandb_conf, reinit=True)
    wandb.run.name = save_name
    wandb.run.save()


train_df = read_data(dataset, "train")
val_df = read_data(dataset, "val")

voc = Vocabulary('BOW', stopwords_lang)
voc.build_vocab(train_df['utterance'].tolist() +  val_df['utterance'].tolist(), max_vocab_size)

if voc.num_words < max_vocab_size or max_vocab_size == 0:
    max_vocab_size = voc.num_words

train_corpus = NLPDataset(dataset, "train", sentence_max_len, vocab= voc)
labels_dict = train_corpus.labels_dict
val_corpus = NLPDataset(dataset, "val", sentence_max_len, labels_dict = labels_dict, vocab= voc)




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

criterion = torch.nn.CrossEntropyLoss()
model = BOWClassifier(train_corpus.num_labels, max_vocab_size, criterion)

#nn.init.normal_(model.classifier.weight.data, 0, 0.02)
#nn.init.zeros_(model.classifier.bias.data)
model = model.to(device)
if debug:
    optimizer = SGD(model.parameters(), lr)
else:
    optimizer = Adam(model.parameters(),lr, betas=(0.9, 0.999))

if log_wandb:
    wandb.watch(model)

n_epochs_no_improvement = 0
gradient_accumulation_steps = 1

t_total = (len(train_dataloader) // gradient_accumulation_steps) * epochs
print(f"====>TOTAL NUMBER OF STEPS: {t_total}")

total_steps = len(train_dataloader) * epochs

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,verbose=True)

training_stats = []

global_step = 0
torch.cuda.empty_cache()

total_t0 = time.time()

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
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: segment_ids
        #   [2]: attention masks
        #   [3]: labels
        bow_vector = batch[0].to(device)
        b_labels = batch[1].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        loss, logits = model(bow_vector,b_labels)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_loss = loss.item()
        if step % 40 == 0 and not step == 0:
            print("  step loss: {0:.2f}".format(step_loss))

        total_train_loss += step_loss
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        global_step += 1
            # Calculate the average loss over all of the batches.
    # print(f"====>SIZE OF TRAIN DATALOADER={len(train_corpus)}")
    avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

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
    for step, batch in enumerate(validation_dataloader):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        bow_vector = batch[0].to(device)
        b_labels = batch[1].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            loss, logits = model(bow_vector,b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss
        batch_acc = get_accuracy_from_logits(logits, b_labels)
        print("Batch accuracy: {0:.2f}".format(batch_acc))
        total_eval_accuracy += batch_acc

    # Report the final accuracy for this validation run.
    # print(f"====> Avg_val_accuracy type={}")
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss /len(validation_dataloader)
    scheduler.step(avg_val_loss)
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
        torch.save(model, join(model_path, "best-model.pth"))
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
        save_path = os.path.join(FULL_PATH_TO_MODELS, f"bow-dataset-{dataset}-{save_name}.pk")
    else:
        save_path = os.path.join(FULL_PATH_TO_MODELS, f"bow-dataset-{dataset}.pk")
    with open(save_path, "wb") as f:
        pk.dump(save_data, f)
