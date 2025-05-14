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

TRAINING_ITERATIONS = int(1.4e5)
PLOT_AFTER = int(1e4)
BATCH_SIZE = 128
MAX_CONTEXT_POINTS = 10
MIN_SIDES = 3
MAX_SIDES = 8
x_size = 4 + 3 * MAX_SIDES
y_size = MAX_SIDES
r_size = 320
torch.manual_seed(0)

criterion = NLLLoss()

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
    max_num_sides=MIN_SIDES,
    center=(5, 5),
    radius=3,
    testing=True,
)

Decoder = partial(
    MLP,
    n_hidden_layers=8,
    hidden_size=r_size,
    dropout=0.1,
    is_res=True,
)

Encoder = partial(
    MLP,
    n_hidden_layers=7,
    hidden_size=r_size,
    dropout=0.1,
    is_res=True,
)

model = CNP(x_dim=x_size, y_dim=y_size, r_dim=r_size, Decoder=Decoder, Encoder=Encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TRAINING_ITERATIONS, eta_min=1e-6
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = next(model.parameters()).device
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

train_losses = []
test_losses  = []
iters_list = []

checkpoint = False

if checkpoint:
    ckpt = torch.load("../models/polygon/np/" + model.__class__.__name__ + "_checkpoint.pt",
                        map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_iter = ckpt['iteration'] + 1
    model.to(device)
    print("Loaded checkpoint from: " + "../models/polygon/np/" + model.__class__.__name__ + "_checkpoint.pt")
else:
    start_iter = 1

# ----------------------
# Training Loop
# ----------------------
for it in range(start_iter, TRAINING_ITERATIONS + 1):
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
    loss = criterion(dist, target_y, mask=context_masks)
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
        loss = criterion(test_dist, target_y_eval, mask=context_masks_eval)
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

        # We average over latent samples.
        pred_target_avg = pred_target.mean(0)  # shape: [batch, n_target, y_dim]

        # print("pred_target_avg shape: ", pred_target_avg.shape)
        pred_target_avg = pred_target_avg[0, -1, :]
        true_polygon = true_poly_eval[0]
        true_polygon_tokenised = true_polygon.to_tokenised()
        # print("pred_target_avg shape after squeeze: ", pred_target_avg.shape)

        print("True Polygon: ", true_polygon)
        print("True Polygon Angles: ", true_polygon.angles)
        print("Predicted Polygon Angles: ", pred_target_avg.tolist()[:len(true_polygon.angles)])

checkpoint = {
    'iteration': it,                                 
    'model_state_dict': model.state_dict(),         
    'optimizer_state_dict': optimizer.state_dict(),  
    'scheduler_state_dict': scheduler.state_dict(),  
}

torch.save(model.state_dict(), "models/polygon/np/angle_completion_task/" + model.__class__.__name__ + ".pt")
torch.save(checkpoint, "models/polygon/np/" + model.__class__.__name__ + "_checkpoint.pt")
print(f"Saved final checkpoint at iteration {it} → models/polygon/np/" + model.__class__.__name__ + "_checkpoint.pt")