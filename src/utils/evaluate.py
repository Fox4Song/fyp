from math import fabs
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import EOS_TOKEN


def evaluate_neural_process_angle_completon_on_batch(
    model, ctx_x, ctx_y, tgt_x, tgt_y, true_polygons, max_seq_len, device
):
    """
    Few-shot angle completion with a Neural Process.
    """
    model.eval()
    with torch.no_grad():
        p_y_dist, _, _, _ = model(ctx_x, ctx_y, tgt_x, tgt_y)
        mean_z = p_y_dist.mean.mean(dim=0)
        pred_y = mean_z.squeeze(1)

    B = pred_y.shape[0]
    all_abs, all_mae, all_sum = [], [], []

    for i in range(B):
        poly = true_polygons[i]
        n = poly.n
        true_angles = poly.angles

        # First n entries of the y-vector are the angles
        pred_vec = pred_y[i].tolist()
        pred_angles = pred_vec[:n]

        # Per-angle absolute errors
        errs = [abs(p - t) for p, t in zip(pred_angles, true_angles)]
        all_abs.extend(errs)

        # Per-polygon MAE
        all_mae.append(sum(errs) / n)

        # Angle-sum error
        all_sum.append(abs(sum(pred_angles) - (n - 2) * 180.0))

    avg_abs_err = sum(all_abs) / len(all_abs)
    avg_angle_mae = sum(all_mae) / len(all_mae)
    avg_angle_sum_err = sum(all_sum) / len(all_sum)

    return avg_abs_err, avg_angle_mae, avg_angle_sum_err


def evaluate_transformer_angle_completion_on_batch(
    model, ctx_x, ctx_y, tgt_x, tgt_y, true_polygons, max_seq_len, device
):
    """
    Few-shot angle completion with a Transformer.
    """
    B, C, _ = ctx_x.shape
    _, _, L_angles = tgt_y.shape

    total_abs_err = 0.0
    total_count = 0
    all_preds = []
    all_trues = []

    model.eval()
    with torch.no_grad():
        for i in range(B):
            # Build prompt
            prompt = []
            for j in range(C):
                toks = ctx_x[i, j].tolist()
                while toks and toks[-1] == 0.0:
                    toks.pop()
                prompt += toks
                toks = ctx_y[i, j].tolist()
                while toks and toks[-1] == 0.0:
                    toks.pop()
                prompt += toks
                prompt.append(EOS_TOKEN)

            partial = tgt_x[i, 0].tolist()
            while partial and partial[-1] == 0.0:
                partial.pop()
            prompt += partial

            prompt_len = len(prompt)

            # Autoregressive loop
            generated = prompt.copy()
            for _ in range(L_angles):
                if len(generated) > max_seq_len:
                    generated = generated[-max_seq_len:]
                inp = (
                    torch.tensor(generated, device=device).unsqueeze(1).unsqueeze(-1)
                )  # [seq_len, 1, 1]
                logits = model(inp)  # → same shape
                nxt = logits[-1, 0, 0].item()
                generated.append(nxt)

            # Slice out predictions
            true_angles = true_polygons[i].angles
            n = true_polygons[i].n
            pred_angles = generated[len(prompt) : len(prompt) + n]

            # Accumulate
            for p, t in zip(pred_angles, true_angles):
                total_abs_err += abs(p - t)
            total_count += n

            all_preds.append(pred_angles)
            all_trues.append(true_angles)

    avg_abs_err = total_abs_err / total_count
    angle_maes = [
        sum(abs(p - t) for p, t in zip(p_list, t_list)) / len(t_list)
        for p_list, t_list in zip(all_preds, all_trues)
    ]
    angle_sum_errs = [
        fabs(sum(p_list) - (len(p_list) - 2) * 180.0) for p_list in all_preds
    ]

    avg_angle_mae = sum(angle_maes) / len(angle_maes)
    avg_angle_sum_err = sum(angle_sum_errs) / len(angle_sum_errs)

    return (
        avg_abs_err,
        avg_angle_mae,
        avg_angle_sum_err,
    )


def compare_models_shared_data(
    reader, np_model, tf_model, tf_max_seq_len, context_sizes, runs_per_size, device
):
    """
    Compare NP and Transformer on same sampled tasks.
    """
    results = {"NP": {}, "TF": {}}

    np_model.to(device).eval()
    tf_model.to(device).eval()

    for c in context_sizes:
        errs_np, maes_np, sums_np = [], [], []
        errs_tf, maes_tf, sums_tf = [], [], []

        for _ in range(runs_per_size):
            # Generate a single batch of tasks (shared data)
            (ctx_x, ctx_y, tgt_x, tgt_y, tokens, true_polys, max_seq_len, _) = (
                reader.generate_polygon_batch_few_shot_completion_task(c)
            )

            ctx_x, ctx_y = ctx_x.to(device), ctx_y.to(device)
            tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)

            # -- Neural Process evaluation --
            avg_np, mae_np, sum_np = evaluate_neural_process_angle_completon_on_batch(
                np_model, ctx_x, ctx_y, tgt_x, tgt_y, true_polys, max_seq_len, device
            )
            errs_np.append(avg_np)
            maes_np.append(mae_np)
            sums_np.append(sum_np)

            # -- Transformer evaluation --
            avg_tf, mae_tf, sum_tf = evaluate_transformer_angle_completion_on_batch(
                tf_model, ctx_x, ctx_y, tgt_x, tgt_y, true_polys, tf_max_seq_len, device
            )
            errs_tf.append(avg_tf)
            maes_tf.append(mae_tf)
            sums_tf.append(sum_tf)

        results["NP"][c] = {
            "abs_err": np.mean(errs_np),
            "angle_mae": np.mean(maes_np),
            "angle_sum": np.mean(sums_np),
        }
        results["TF"][c] = {
            "abs_err": np.mean(errs_tf),
            "angle_mae": np.mean(maes_tf),
            "angle_sum": np.mean(sums_tf),
        }

    # Plot results
    for metric, ylabel in [
        ("abs_err", "Average Absolute Angle Error"),
        ("angle_mae", "Angle MAE"),
        ("angle_sum", "Angle Sum Error"),
    ]:
        plt.figure(figsize=(7, 4))
        for name, marker in [("NP", "-o"), ("TF", "-s")]:
            ys = [results[name][c][metric] for c in context_sizes]
            plt.plot(context_sizes, ys, marker, label=name)
        plt.xlabel("Number of Context Points")
        plt.ylabel(ylabel)
        plt.xticks(context_sizes)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
