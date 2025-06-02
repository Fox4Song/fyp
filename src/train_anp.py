from data import PolygonSentenceReader, Polygon
from modules import NLLLoss, ELBOLoss, MLP
from neural_process.models.np import CNP, LNP
from neural_process.models.attnnp import AttnCNP, AttnLNP
from transformer.models import TransformerDecoder
from utils import plot_polygon

import datetime
from functools import partial
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

TRAINING_ITERATIONS = int(1e5)
PLOT_AFTER = int(5e3)
BATCH_SIZE = 128
MAX_CONTEXT_POINTS = 15
MIN_SIDES = 3
MAX_SIDES = 12
x_size = 4 + 3 * MAX_SIDES
y_size = 12
torch.manual_seed(0)

criterion = ELBOLoss()

# Instantiate a polygon generator.
# (For example, polygons with between 3 and 8 sides.)
polygon_generator_train = PolygonSentenceReader(
    batch_size=BATCH_SIZE,
    max_num_context=MAX_CONTEXT_POINTS,
    max_seq_len=x_size,
    min_num_sides=MIN_SIDES,
    max_num_sides=MAX_SIDES,
    center=(5, 5),
    radius=3,
    testing=False,
)

polygon_generator_test = PolygonSentenceReader(
    batch_size=100,
    max_num_context=MAX_CONTEXT_POINTS,
    max_seq_len=x_size,
    min_num_sides=MIN_SIDES,
    max_num_sides=MAX_SIDES,
    center=(5, 5),
    radius=3,
    testing=True,
)

Encoder = partial(
    MLP,
    n_hidden_layers=2,
    hidden_size=224,
)

Decoder = partial(
    MLP,
    n_hidden_layers=5,
    hidden_size=224,
    dropout=0.1,
)

LatentEncoder = partial(
    MLP,
    n_hidden_layers=3,
    hidden_size=224,
    dropout=0.1,
)

model = AttnLNP(
    x_dim=x_size,
    y_dim=y_size,
    r_dim=224,
    attention_type="multihead",
    n_z_train=10,
    n_z_test=10,
    Encoder=Encoder,
    LatentEncoder=LatentEncoder,
    Decoder=Decoder,
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TRAINING_ITERATIONS, eta_min=1e-6
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = next(model.parameters()).device
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

train_losses = []
test_losses = []
iters_list = []

# ----------------------
# Training Loop
# ----------------------
for it in range(1, TRAINING_ITERATIONS + 1):
    # Generate a batch of polygon completion tasks.
    (
        context_x,
        context_y,
        target_x,
        target_y,
        tokens,
        true_poly,
        max_seq_len,
        num_context,
        num_target,
        context_masks,
    ) = polygon_generator_train.generate_polygon_batch_few_shot_completion_task()

    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    context_masks = context_masks.to(device)

    optimizer.zero_grad()
    # Forward pass: the NP model expects context_x, context_y, target_x, target_y.
    dist, z, q_zc, q_zct = model(context_x, context_y, target_x, target_y)
    loss = criterion(dist, q_zct, q_zc, target_y, mask=context_masks)
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(loss.item())

    if it % 1000 == 0:
        print("Iteration: {}, train loss: {}".format(it, loss.item()))

    # ----------------------
    # Evaluation and Plotting
    # ----------------------
    if it % PLOT_AFTER == 0:
        # For plotting, we generate a single polygon sample.
        (
            context_x_eval,
            context_y_eval,
            target_x_eval,
            target_y_eval,
            tokens_eval,
            true_poly_eval,
            max_seq_len_eval,
            num_context_eval,
            num_target_eval,
            context_masks_eval,
        ) = polygon_generator_test.generate_polygon_batch_few_shot_completion_task()

        context_x_eval = context_x_eval.to(device)
        context_y_eval = context_y_eval.to(device)
        target_x_eval = target_x_eval.to(device)
        target_y_eval = target_y_eval.to(device)
        context_masks_eval = context_masks_eval.to(device)

        # Forward pass through the model.
        test_dist, test_z, test_q_zc, test_q_zct = model(
            context_x_eval, context_y_eval, target_x_eval, target_y_eval
        )
        loss = criterion(
            test_dist, test_q_zct, test_q_zc, target_y_eval, mask=context_masks_eval
        )
        test_losses.append(loss.item())
        iters_list.append(it)

        print(
            "{}, Iteration: {}, Test Loss: {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                it,
                loss.item(),
            )
        )

        # Get NP predictions on target tokens (taking the mean of the predictive distribution).
        pred_target = test_dist.mean  # shape: [n_z, batch, n_target, y_dim]

        # For simplicity, we average over latent samples and batch dimension.
        pred_target_avg = pred_target.mean(0)[
            0, -1, :
        ]  # shape: [batch, n_target, y_dim]
        true_polygon = true_poly_eval[0]
        true_polygon_tokenised = true_polygon.to_tokenised()
        pred_target_avg = pred_target_avg[: len(true_polygon.angles)].tolist()

        print("True Polygon: ", true_polygon)
        print("True Polygon Angles: ", true_polygon.angles)
        print("Predicted Polygon Angles: ", pred_target_avg)

checkpoint = {
    "iteration": it,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}

torch.save(
    model.state_dict(),
    "models/polygon/np/angle_completion_task/" + model.__class__.__name__ + ".pt",
)
torch.save(
    checkpoint, "models/polygon/np/" + model.__class__.__name__ + "_checkpoint.pt"
)
print(
    f"Saved final checkpoint at iteration {it} → models/polygon/np/"
    + model.__class__.__name__
    + "_checkpoint.pt"
)
