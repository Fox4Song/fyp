from data import PolygonSentenceReader
from modules import MSELoss
from transformer.models import TransformerDecoder

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

# --- Set up data generation parameters ---
TRAINING_ITERATIONS = int(3e5)
WARMUP_ITERATIONS = int(1e4)
BATCH_SIZE = 128
CLIP_NORM = 1.0
LOG_INTERVAL = 1000
INITIAL_LR = 1e-4
MIN_LR = 1e-6

max_seq_len = 512
min_num_sides = 3
max_num_sides = 12
max_num_context = 10  # dummy for reader

criterion = MSELoss()

polygon_reader = PolygonSentenceReader(
    batch_size=BATCH_SIZE,
    max_num_context=max_num_context,
    max_seq_len=max_seq_len,
    min_num_sides=min_num_sides,
    max_num_sides=max_num_sides,
    testing=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & optimizer ---
x_dim, y_dim = 1, 1
model = TransformerDecoder(
    x_dim, y_dim, r_dim=256, decoder_layers=8, decoder_heads=8
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-2)

warmup_scheduler = LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_ITERATIONS
)

cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=TRAINING_ITERATIONS - WARMUP_ITERATIONS, eta_min=MIN_LR
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_ITERATIONS],
)

# --- Training Loop ---
model.train()
running_loss = 0.0
for it in range(TRAINING_ITERATIONS + 1):
    # 1) grab a causal batch: input, next-token labels, and padding mask
    #    shapes: [B, L], [B, L], [B, L]
    input_batch, label_batch, attention_mask_batch = (
        polygon_reader.generate_causal_polygon_batch()
    )

    # 2) move to device
    input_batch = input_batch.to(device)  # [B, L]
    label_batch = label_batch.to(device)  # [B, L]
    attention_mask_batch = attention_mask_batch.to(device)  # [B, L]

    # 3) reshape for PyTorch Transformer: [seq_len, batch, x_dim]
    #    here x_dim=1 so we unsqueeze a feature dim
    src = input_batch.unsqueeze(-1).permute(1, 0, 2)  # [L, B, 1]
    padding_mask = ~(attention_mask_batch.to(device).bool())  # [B, L]

    optimizer.zero_grad()
    # 4) forward pass through decoder-only stack
    #    returns logits/predictions of shape [L, B, y_dim]
    out = model(src, src_key_padding_mask=padding_mask)  # [L, B, 1]

    # 5) reshape back to [B, L]
    preds = out.permute(1, 0, 2).squeeze(-1)  # [B, L]

    # 6) compute masked MSE loss (ignore padding positions)
    #    we weight squared error by attention_mask (1. on real tokens, 0. on pad)
    loss = criterion(preds, label_batch, attention_mask_batch)
    running_loss += loss.item()

    # 7) backward + step
    loss.backward()
    clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()
    scheduler.step()

    if it % LOG_INTERVAL == 0 and it > 0:
        avg_loss = running_loss / LOG_INTERVAL
        lr = scheduler.get_last_lr()[0]
        print(
            f"[{it:6d}/{TRAINING_ITERATIONS}] " f"lr={lr:.2e}  avg_loss={avg_loss:.4f}"
        )
        running_loss = 0.0

final_checkpoint_path = f"final_tf_model_{total_params}_params.pt"
torch.save(
    {
        "iteration": TRAINING_ITERATIONS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss.item(),
    },
    final_checkpoint_path,
)
print("Training complete. Final model checkpoint saved to", final_checkpoint_path)
