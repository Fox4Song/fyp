import argparse
import datetime
from functools import partial
import torch
from torch.distributions.kl import kl_divergence
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import PolygonSentenceReader, Polygon
from modules import NLLLoss, ELBOLoss, MLP
from neural_process.models.np import CNP, LNP
from neural_process.models.attnnp import AttnLNP
from utils import plot_polygon


def get_model_config(model_name, x_size, y_size):
    if model_name == 'lnp':
        return {
            'model_cls': LNP,
            'model_kwargs': {'x_dim': x_size, 'y_dim': y_size, 'r_dim': 256, 'n_z_train': 10, 'n_z_test': 10},
            'criterion': ELBOLoss(),
            'lr': 1e-4,
            'max_context': 15,
            'batch_size': 128,
        }
    elif model_name == 'cnp':
        r_size = 320
        return {
            'model_cls': CNP,
            'model_kwargs': {
                'x_dim': x_size, 'y_dim': y_size, 'r_dim': r_size,
                'Decoder': partial(MLP, n_hidden_layers=8, hidden_size=r_size, dropout=0.1, is_res=True),
                'Encoder': partial(MLP, n_hidden_layers=7, hidden_size=r_size, dropout=0.1, is_res=True),
            },
            'criterion': NLLLoss(),
            'lr': 1e-4,
            'max_context': 15,
            'batch_size': 128,
        }
    elif model_name == 'anp':
        return {
            'model_cls': AttnLNP,
            'model_kwargs': {
                'x_dim': x_size, 'y_dim': y_size, 'r_dim': 224,
                'attention_type': 'multihead', 'n_z_train': 10, 'n_z_test': 10,
                'Encoder': partial(MLP, n_hidden_layers=2, hidden_size=224),
                'LatentEncoder': partial(MLP, n_hidden_layers=3, hidden_size=224, dropout=0.1),
                'Decoder': partial(MLP, n_hidden_layers=5, hidden_size=224, dropout=0.1),
            },
            'criterion': ELBOLoss(),
            'lr': 5e-5,
            'max_context': 15,
            'batch_size': 128,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def transform_train(model_name, iters, plot_after, device, resume):
    MIN_SIDES, MAX_SIDES = 3, 8
    x_size = 5 + 4 * MAX_SIDES
    y_size = x_size

    cfg = get_model_config(model_name, x_size, y_size)
    BATCH = cfg['batch_size']
    MAX_CTX = cfg['max_context']

    train_gen = PolygonSentenceReader(
        batch_size=BATCH, max_num_context=MAX_CTX, max_seq_len=x_size,
        min_num_sides=MIN_SIDES, max_num_sides=MAX_SIDES,
        center=(5,5), radius=3, testing=False)
    test_gen = PolygonSentenceReader(
        batch_size=100, max_num_context=MAX_CTX, max_seq_len=x_size,
        min_num_sides=MIN_SIDES, max_num_sides=MIN_SIDES,
        center=(5,5), radius=3, testing=True)

    model = cfg['model_cls'](**cfg['model_kwargs']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    sch = CosineAnnealingLR(opt, T_max=iters, eta_min=1e-6)

    start = 1
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sch.load_state_dict(ckpt['scheduler_state_dict'])
        start = ckpt['iteration'] + 1
        print(f"Resumed from {resume} at iteration {start}")

    train_losses, test_losses, iters_list = [], [], []

    for it in range(start, iters+1):
        ctx_x, ctx_y, tgt_x, tgt_y, tokens, true_poly, _, _, _, ctx_mask = \
            train_gen.generate_polygon_batch_few_shot_masked_completion_task()
        ctx_x, ctx_y, tgt_x, tgt_y, ctx_mask = [t.to(device) for t in (ctx_x, ctx_y, tgt_x, tgt_y, ctx_mask)]

        opt.zero_grad()
        dist, _, q_zc, q_zct = model(ctx_x, ctx_y, tgt_x, tgt_y)
        if model_name == "cnp":
            loss = cfg['criterion'](dist, tgt_y, mask=ctx_mask)
        else:
            loss = cfg['criterion'](dist, q_zct, q_zc, tgt_y, mask=ctx_mask)
        loss.backward()
        opt.step()
        sch.step()
        train_losses.append(loss.item())

        if it % 1000 == 0:
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}, Iteration: {it}, Train Loss: {loss.item():.4f}")

        if it % plot_after == 0:
            cx, cy, tx, ty, tok_e, tp, _, _, _, cx_mask = \
                test_gen.generate_polygon_batch_few_shot_masked_completion_task()
            cx, cy, tx, ty, cx_mask = [t.to(device) for t in (cx, cy, tx, ty, cx_mask)]

            td, _, qzc, qzct = model(cx, cy, tx, ty)
            if model_name == "cnp":
                t_loss = cfg['criterion'](td, ty, mask=cx_mask)
            else:
                t_loss = cfg['criterion'](td, qzct, qzc, ty, mask=cx_mask)
            test_losses.append(t_loss.item()); iters_list.append(it)

            pred = td.mean.mean(0)[0, -1, :]
            true_orig = tp[0][0]
            tok_orig = true_orig.to_tokenised()
            predicted = pred[:len(tok_orig)].tolist()

            mask_1d = cx_mask.squeeze(1)[0].cpu().tolist()
            mask_1d = mask_1d[: len(tok_orig)]

            # Find which positions were masked
            masked_positions = [idx for idx, m in enumerate(mask_1d) if m == 1]

            true_vals = [tok_orig[idx] for idx in masked_positions]
            pred_vals = [predicted[idx] for idx in masked_positions]

            print(f"Iteration {it}:")
            print("Test loss: ", t_loss.item())
            print("Masked positions:      ", masked_positions)
            print("True masked values:    ", true_vals)
            print("Predicted masked vals: ", pred_vals)

    # Save final model and checkpoint
    base = f"models/polygon/np/masked_completion_task/{cfg['model_cls'].__name__}"
    ck = {
        'iteration': it, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': sch.state_dict()
    }
    torch.save(model.state_dict(), f"{base}.pt")
    torch.save(ck, f"{base}_checkpoint.pt")
    print(f"Saved masked completion model and checkpoint at iteration {it}")


if __name__ == '__main__':
    p = argparse.ArgumentParser("Train NP completion models")
    p.add_argument('--model', choices=['lnp','cnp','anp'], required=True)
    p.add_argument('--iters', type=int, default=80000)
    p.add_argument('--plot_after', type=int, default=10000)
    p.add_argument('--resume', type=str, help='checkpoint path')
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_train(args.model, args.iters, args.plot_after, dev, args.resume)
