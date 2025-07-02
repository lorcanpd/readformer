import torch

from tensordict.nn import TensorDictModule
from torchrl.data import OneHot
from torchrl.objectives import DQNLoss
from torchrl.modules import QValueActor
from torchrl.data.replay_buffers import (
    TensorDictPrioritizedReplayBuffer,
    LazyMemmapStorage,
)

import os
from shutil import rmtree

def alpha_beta_to_q(alpha, beta, V):
    p = alpha / (alpha + beta + 1e-8)
    A1 = p - 0.5
    A0 = -A1
    q0 = V + A0
    q1 = V + A1
    return torch.stack([q0, q1], dim=-1)  # (N, 2)


def build_value_head(args, device):
    """A tiny MLP V-head f(h) -> R used by the Q-network."""
    return torch.nn.Sequential(
        torch.nn.Linear(args.emb_dim, args.emb_dim // 2, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(args.emb_dim // 2, 1, bias=True)
    ).to(device)


class QNet(torch.nn.Module):
    """
    A *thin* wrapper: keeps separate references to
      • embedder (InputEmbeddingLayer)
      • backbone (Model / ReadFormer)
      • classifier (BetaDistributionClassifier)
      • value_head (small MLP that outputs V(s))
      • ref_emb (optional NucleotideEmbeddingLayer)

    Forward expects a **TensorDict** containing the *raw* tensors that
    already exist in your batch / replay-buffer.
    """

    def __init__(self, embedder, backbone, classifier, value_head, ref_emb):
        super().__init__()
        self.embedder = embedder  # <- unchanged modules
        self.backbone = backbone
        self.classifier = classifier
        self.value_head = value_head
        self.ref_emb = ref_emb  # can be None

    def forward(self, td):  # td: TensorDict with keys below
        ns = td["nucleotide_sequences"]
        bq = td["base_qualities"]
        ce = td["cigar_encoding"]
        isf = td["is_first"]
        m2r = td["mapped_to_reverse"]
        pos = td["positions"]
        ref = td.get("reference", None)

        x = self.embedder(ns, ce, bq, m2r, isf)  # (B,L,D)
        h = self.backbone(x, pos)  # (B,L,D)
        # keep only the embedding which corresponds to our candidate mutation
        idx = torch.nonzero(pos == td["mut_pos"], as_tuple=True)
        h = h[idx]  # (B,L,D) -> (B,D)
        if self.ref_emb is not None and ref is not None:
            ref = self.ref_emb(ref).squeeze(-2)  # (B,D)
        alpha, beta = self.classifier(h, ref)  # (B,L,1)
        alpha, beta = alpha.squeeze(-1), beta.squeeze(-1)  # (B,L)
        V = self.value_head(h).squeeze(-1)  # (B,L)
        q = alpha_beta_to_q(alpha, beta, V)
        return q


def instantiate_rl_model(
        args, device, input_emb, model, classifier, ref_emb=None,
        gamma=0.8
):
    """
    Create the parts of the RL model that aren't in the base model.

    Returns the QNet instance, the value head, and the DQNLoss.
    """

    value_head = build_value_head(args, device)

    q_model = QNet(
        input_emb, model, classifier, value_head, ref_emb
    ).to(device)

    in_keys = [
        "observation"
    ]

    q_td_module = TensorDictModule(
        module=q_model,
        in_keys=in_keys,
        out_keys=["action_value"],
    )

    spec = OneHot(2)

    actor = QValueActor(
        q_td_module,
        in_keys=["nucleotide_sequences", "base_qualities", "cigar_encoding",
                 "is_first", "mapped_to_reverse", "positions", "reference"],
        spec=spec
    ).to(device)

    dqn_loss = DQNLoss(
        actor,
        delay_value=True, double_dqn=True,
        action_space=spec
    )

    dqn_loss.make_value_estimator(dqn_loss.value_type, gamma=gamma)

    return q_model, value_head, dqn_loss


def clear_dir(path: str) -> None:
    """
    Delete *all* files and subdirectories in `path`,
    but leave `path` itself intact.
    """
    for name in os.listdir(path):
        full = os.path.join(path, name)
        # unlink files & symlinks, rmtree directories
        if os.path.isdir(full):
            rmtree(full)
        else:
            os.unlink(full)


def instantiate_replay_buffers(args):
    """
    Create the balanced replay buffers for positive and negative pools.
    """

    for pool in ("positive_memmap", "negative_memmap"):
        d = os.path.join(args.scratch_dir, pool)
        os.makedirs(d, exist_ok=True)  # create if missing
        clear_dir(d)  # now fully empty it

    # positive pool
    storage_pos = LazyMemmapStorage(
        max_size=500_000,
        scratch_dir=os.path.join(args.scratch_dir, "positive_memmap"),
        existsok=True,
    )
    rb_pos = TensorDictPrioritizedReplayBuffer(
        alpha=0.6, beta=0.4,
        storage=storage_pos,
        batch_size=args.batch_size // 2,
    )

    # negative pool
    storage_neg = LazyMemmapStorage(
        max_size=500_000,
        scratch_dir=os.path.join(args.scratch_dir, "negative_memmap"),
        existsok=True,
    )
    rb_neg = TensorDictPrioritizedReplayBuffer(
        alpha=0.6, beta=0.4,
        storage=storage_neg,
        batch_size=args.batch_size // 2,
    )

    return rb_pos, rb_neg