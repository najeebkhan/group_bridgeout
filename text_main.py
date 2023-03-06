"""
Code is based on PyTorch Transformer tutorial

https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import time
import math
from utils import comet_experiment as experiment


import torch
from torch import nn, Tensor
import numpy as np
from utils import TransformerModel, IterMeter
from utils import generate_square_subsequent_mask, get_data, get_batch, get_model_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
#                   CONFIG, DATA and MODEL
# -------------------------------------------------------------


class Config:
    def __init__(self) -> None:

        self.batch_size = 20

        self.bptt = 70
        self.emsize = 1024  # embedding dimension
        self.d_hid = 2048  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 4  # number of heads in nn.MultiheadAttention
        self.dropout = 0.5
        self.pos_enc_droput = 0.2
        self.gbo = 0.3

        self.lr = 5.  # learning rate
        self.lr_milestones = [10, 100, 150, 230]
        self.lr_decay_by = 0.25

        self.epochs = 300


CFG = Config()
train_data, val_data, test_data, vocab = get_data(CFG.batch_size)
ntokens = len(vocab)

model = TransformerModel(ntokens,
                         CFG.emsize,
                         CFG.nhead,
                         CFG.d_hid,
                         CFG.nlayers,
                         CFG.dropout,
                         CFG.pos_enc_droput,
                         CFG.gbo).to(device)


# -------------------------------------------------------------
#           Training and Evaluation Utilities
# -------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    CFG.lr_milestones,
    gamma=CFG.lr_decay_by)


def train(model: nn.Module, experiment, iter_meter, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(CFG.bptt).to(device)
    num_batches = len(train_data) // CFG.bptt
    with experiment.train():
        for batch, i in enumerate(range(0, train_data.size(0) - 1, CFG.bptt)):
            data, targets = get_batch(train_data, i, CFG.bptt)
            seq_len = data.size(0)
            if seq_len != CFG.bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            iter_meter.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(min(700, cur_loss))

                experiment.log_metric('ppl', ppl, step=iter_meter.get())
                experiment.log_metric(
                    'learning_rate', scheduler.get_lr(), step=iter_meter.get())
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0

                start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor, experiment, epoch) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(CFG.bptt).to(device)
    with experiment.test():
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, CFG.bptt):
                data, targets = get_batch(eval_data, i, CFG.bptt)
                seq_len = data.size(0)
                if seq_len != CFG.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += seq_len * criterion(output_flat, targets).item()
        experiment.log_metric('val_loss', total_loss /
                              (len(eval_data) - 1), step=epoch)
        experiment.log_metric('val_ppl', np.exp(
            total_loss / (len(eval_data) - 1)), step=epoch)
    return total_loss / (len(eval_data) - 1)


# -------------------------------------------------------------
#                           Main Loop
# -------------------------------------------------------------

def main():
    best_val_loss = float('inf')

    best_model_params_path = 'models/' + get_model_path(vars(CFG))

    
    experiment.log_parameters(vars(CFG))

    iter_meter = IterMeter()

    for epoch in range(1, CFG.epochs + 1):
        epoch_start_time = time.time()
        train(model, experiment, iter_meter, epoch)
        val_loss = evaluate(model, val_data, experiment, epoch)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)
                          )  # load best model states

    test_loss = evaluate(model, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

    experiment.end()


if __name__ == '__main__':
    # todo: use argparse to modify config and pass to main
    main()
