import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import math
from contextlib import nullcontext
import numpy as np
import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import PretrainDataset
import logging

# To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_model(path):
    if ddp:
        if torch.distributed.get_rank() == 0:
            torch.save(raw_model.state_dict(), path)
    else:
        torch.save(raw_model.state_dict(), path)


@torch.no_grad()
def evaluate(val_loader):
    model.eval()
    losses = []
    for batch_idx, (X, Y) in enumerate(val_loader):
        if batch_idx >= eval_iters:
            break
        X = X.to(device)
        Y = Y.to(device)
        with ctx:
            _ = model(X, Y)
            loss = raw_model.last_loss
        losses.append(loss.detach())

    if len(losses) == 0:
        mean_loss = torch.tensor(float("inf"), device=device)
    else:
        mean_loss = torch.stack(losses).mean()

    if ddp:
        # Average validation loss across ranks
        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
        mean_loss = mean_loss / ddp_world_size

    model.train()
    return mean_loss.item()


def should_stop_training(no_improve_count):
    if not ddp:
        return no_improve_count >= patience

    stop_tensor = torch.tensor(
        1 if no_improve_count >= patience and master_process else 0,
        device=device,
        dtype=torch.int32,
    )
    torch.distributed.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


def train_epoch(epoch, global_step, best_val_loss, no_improve_count):
    start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)

        lr = get_lr(global_step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if ddp:
            model.require_backward_grad_sync = 0 == gradient_accumulation_steps - 1

        with ctx:
            _ = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_interval == 0 and master_process:
                spend_time = time.time() - start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) step:{} loss:{:.4f} lr:{:.7f} ETA:{}min'.format(
                        epoch,
                        max_epoch,
                        step,
                        iter_per_epoch,
                        global_step,
                        loss.item() * gradient_accumulation_steps,
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    )
                )

            if global_step % save_interval == 0:
                save_model('{}/iter_{}.pth'.format(save_dir, global_step))

            if global_step % eval_interval_steps == 0:
                val_loss = evaluate(val_loader)
                if master_process:
                    logger.info(
                        'Eval step:{} val_loss:{:.6f} best_val_loss:{:.6f} no_improve_count:{}/{}'.format(
                            global_step, val_loss, best_val_loss, no_improve_count, patience
                        )
                    )

                    if val_loss < (best_val_loss - min_delta):
                        best_val_loss = val_loss
                        no_improve_count = 0
                        save_model('{}/best.pth'.format(save_dir))
                        logger.info('New best model saved at step {} with val_loss {:.6f}'.format(global_step, val_loss))
                    else:
                        no_improve_count += 1

                if should_stop_training(no_improve_count):
                    if master_process:
                        logger.info('Early stopping triggered at step {}'.format(global_step))
                    return global_step, best_val_loss, no_improve_count, True

    return global_step, best_val_loss, no_improve_count, False


def init_model():
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    if init_from == "scratch":
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    out_dir = 'out'
    max_epoch = 10

    # validation / early stopping
    val_ratio = 0.03
    eval_interval_steps = 1000
    eval_iters = 200
    patience = 8
    min_delta = 1e-4

    log_interval = 100
    save_interval = 10000
    init_from = 'scratch'

    gradient_accumulation_steps = 1
    batch_size = 32

    max_seq_len = 512
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.0
    bias = False

    # anti-plateau toggles
    use_random_sampling_train = True
    use_random_sampling_val = False

    learning_rate = 3e-4
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 80000
    min_lr = 1e-5

    backend = 'nccl'
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = False

    save_dir = os.path.join(out_dir, 'pretrain')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir, 'log.log'))

    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        if os.name == 'nt':
            init_process_group(backend="gloo")
        else:
            init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=ptdtype)

    best_val_loss = 1e9
    no_improve_count = 0
    global_step = 0

    data_path_list = [
        './data/merged_multilingual_zh1_en1_nl1_100m.bin'
    ]

    # split by token range on the same 100M bin
    with open(data_path_list[0], 'rb') as f:
        f.seek(0, 2)
        total_tokens = f.tell() // np.dtype('uint16').itemsize

    val_tokens = max(max_seq_len + 1, int(total_tokens * val_ratio))
    train_tokens = total_tokens - val_tokens
    if train_tokens <= max_seq_len:
        raise ValueError("val_ratio is too large, no data left for training")

    # align boundary for cleaner chunking
    train_tokens = (train_tokens // max_seq_len) * max_seq_len
    if train_tokens <= max_seq_len:
        raise ValueError("train token range too small after alignment")

    train_ds = PretrainDataset(
        data_path_list,
        max_length=max_seq_len,
        memmap=True,
        random_sampling=use_random_sampling_train,
        start_token=0,
        end_token=train_tokens,
        seed=1337 + seed_offset,
    )
    val_ds = PretrainDataset(
        data_path_list,
        max_length=max_seq_len,
        memmap=True,
        random_sampling=use_random_sampling_val,
        start_token=train_tokens,
        end_token=total_tokens,
        seed=1337,
    )

    if master_process:
        logger.info(
            f"Dataset split done. total_tokens={total_tokens}, train_tokens={train_tokens}, "
            f"val_tokens={total_tokens - train_tokens}, val_ratio={val_ratio}, "
            f"random_train={use_random_sampling_train}, random_val={use_random_sampling_val}"
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if ddp else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=train_sampler is None,
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        sampler=val_sampler,
    )

    model = init_model()
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    iter_per_epoch = len(train_loader)

    # Initial validation before training
    initial_val_loss = evaluate(val_loader)
    if master_process:
        logger.info('Initial val_loss: {:.6f}'.format(initial_val_loss))
        best_val_loss = initial_val_loss
        save_model('{}/best.pth'.format(save_dir))

    early_stopped = False
    for epoch in range(max_epoch):
        if ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        global_step, best_val_loss, no_improve_count, should_stop = train_epoch(
            epoch, global_step, best_val_loss, no_improve_count
        )

        save_model('{}/epoch_{}.pth'.format(save_dir, epoch))

        if should_stop:
            early_stopped = True
            break

    save_model('{}/last.pth'.format(save_dir))

    if master_process:
        if early_stopped:
            logger.info('Training finished with early stopping. best_val_loss={:.6f}'.format(best_val_loss))
        else:
            logger.info('Training finished all epochs. best_val_loss={:.6f}'.format(best_val_loss))

    if ddp:
        destroy_process_group()
