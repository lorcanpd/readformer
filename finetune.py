import argparse
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torchrl.objectives.utils import SoftUpdate
from tensordict import TensorDict
from itertools import chain

import os
import sys

import numpy as np

import logging
import wandb

from collections import deque
import multiprocessing as mp

from components.base_model import Model
from components.read_embedding import InputEmbeddingLayer, NucleotideEmbeddingLayer
from components.finetune_data_streaming import create_finetuning_dataloader
from components.classification_head import BetaDistributionClassifier
from components.utils import get_effective_number, get_layerwise_param_groups
from components.metrics import (
    BetaBernoulliLoss, FineTuningMetrics, BalancedPUQuantileLoss,
    run_validation, run_validation_rl, pr_auc_at_prior
)
from components.ddqn import instantiate_rl_model, instantiate_replay_buffers
from pretrain_readwise_only import device_context, check_cuda_availability

import gc


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune with 3 phases")
    # ... all your original arguments, unchanged ...
    parser.add_argument('--name', type=str, required=True, help='Name')
    parser.add_argument('--project', type=str, default='readformer_finetuning')
    parser.add_argument('--emb_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--n_order', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--num_hyena', type=int, default=1)
    parser.add_argument('--num_attention', type=int, default=2)
    parser.add_argument('--readformer', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pre_trained_path', type=str)
    parser.add_argument('--finetune_save_dir', type=str, required=True)
    parser.add_argument('--finetune_metadata_dir', type=str, required=True)
    parser.add_argument('--mutation_bam_path', type=str, required=True)
    parser.add_argument('--artefact_bam_path', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max_read_length', type=int, default=151)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_api_path', type=str, default='.wandb_api')
    parser.add_argument('--load_latest_checkpoint', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--phases_per_epoch', type=int, default=1)
    parser.add_argument('--burn_in_iters', type=int, default=0)
    parser.add_argument('--max_base_quality', type=int, default=50)
    parser.add_argument('--no_reference', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--alpha_prior', type=float, default=1.0)
    parser.add_argument('--beta_prior', type=float, default=1.0)
    parser.add_argument('--use_RL', action='store_true',
                        help='Switch to DQN-style reinforcement learning.')
    parser.add_argument('--scratch_dir', type=str, default='/home',
                        help='Directory for storing temporary files.')
    # parser.add_argument('--phase_three_only', action='store_true')
    # parser.add_argument('--classifier_only', action='store_true')
    # parser.add_argument(
    #     '--phase3_warmup',
    #     type=int,
    #     default=5000,
    #     help='Number of iterations in phase 3 before counting patience'
    # )
    return parser.parse_args()


def instantiate_model(args, device):
    input_embedding = InputEmbeddingLayer(args.emb_dim, args.max_base_quality).to(device)
    readformer_model = Model(
        emb_dim=args.emb_dim, heads=args.num_heads, num_layers=args.num_layers,
        n_order=args.n_order, readformer=args.readformer,
        kernel_size=args.kernel_size, num_hyena=args.num_hyena,
        num_attention=args.num_attention
    ).to(device)
    return input_embedding, readformer_model


def load_pretrained_model(args, device):
    input_embedding, readformer_model = instantiate_model(args, device)
    if not os.path.isfile(args.pre_trained_path):
        logging.error(f"No checkpoint at '{args.pre_trained_path}'")
        sys.exit(1)
    ckpt = torch.load(args.pre_trained_path, map_location=device)
    input_embedding.load_state_dict(ckpt['input_embedding_state_dict'])
    readformer_model.load_state_dict(ckpt['model_state_dict'])
    logging.info(f"Loaded pre-trained model from '{args.pre_trained_path}'")
    return input_embedding, readformer_model


def load_latest_checkpoint(args, device):
    if not os.path.isdir(args.finetune_save_dir):
        logging.error(f"No dir at '{args.finetune_save_dir}'")
        sys.exit(1)
    files = sorted(
        [f for f in os.listdir(args.finetune_save_dir) if f.endswith('.pth')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    latest = os.path.join(args.finetune_save_dir, files[-1])
    ckpt = torch.load(latest, map_location=device)
    input_embedding, readformer_model = instantiate_model(args, device)
    if not args.no_reference:
        ref_base_embedding = NucleotideEmbeddingLayer(
            args.emb_dim, mlm_mode=True).to(device)
        ref_base_embedding.load_state_dict(ckpt['ref_base_embedding_state_dict'])
    else:
        ref_base_embedding = None
    classifier = BetaDistributionClassifier(input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2).to(device)
    input_embedding.load_state_dict(ckpt['input_embedding_state_dict'])
    readformer_model.load_state_dict(ckpt['model_state_dict'])
    classifier.load_state_dict(ckpt['classifier_state_dict'])

    value_head = q_model = dqn_loss = None
    epsilon = 1.0

    if args.use_RL:
        q_model, value_head, dqn_loss = instantiate_rl_model(
            args, device, input_embedding, readformer_model, classifier,
            ref_emb=ref_base_embedding
        )

        # overwrite the target-net weights *that DQNLoss created*
        dqn_loss.target_value_network_params.load_state_dict(ckpt["target_params"])

        epsilon = ckpt.get("epsilon", 1.0)

    min_lr = args.base_lr / 3
    param_groups = (
            get_layerwise_param_groups(readformer_model, args.base_lr, min_lr)
            + [{"params": input_embedding.parameters(), "lr": min_lr}]
            + [
                {"params": (
                        [x for x in classifier.parameters()] +
                        ([] if ref_base_embedding is None
                         else list(ref_base_embedding.parameters())) +
                        ([] if value_head is None
                         else list(value_head.parameters()))),
                    "lr": args.base_lr}
            ]
    )
    optimiser = AdamW(
        param_groups, eps=1e-9, weight_decay=0.01
    )
    optimiser.load_state_dict(ckpt['optimiser_state_dict'])
    epoch, iteration = ckpt['epoch'], ckpt['iteration']
    logging.info(f"Loaded checkpoint '{latest}'")
    return (
        input_embedding, readformer_model, ref_base_embedding, classifier,
        optimiser, q_model, dqn_loss, value_head, epsilon,
        epoch, iteration
    )


def get_allocated_cpus():
    cpus = int(os.getenv('LSB_DJOB_NUMPROC', '1'))
    logging.info(f"Allocated CPUs: {cpus}")
    return cpus


def unfreeze_layers_by_epoch(param_groups, epoch, ignore_groups=[]):
    for i, g in enumerate(param_groups):
        req = (i < epoch) or (i in ignore_groups)
        for p in g['params']:
            p.requires_grad = req


def save_best_checkpoint(
        args, input_embedding, readformer_model, classifier, optimiser, epoch,
        iteration, ref_base_embedding=None, additional_string=None,
        value_head=None, q_model=None, epsilon=None,
        dqn_loss=None
):
    if additional_string is None:
        path = f"{args.finetune_save_dir}/phase_000.pth"
    else:
        path = f"{args.finetune_save_dir}/phase_000_{additional_string}.pth"
    ckpt = {
        'input_embedding_state_dict': input_embedding.state_dict(),
        'model_state_dict': readformer_model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }
    if ref_base_embedding is not None:
        ckpt['ref_base_embedding_state_dict'] = ref_base_embedding.state_dict()

    if value_head is not None:
        ckpt["value_head"] = value_head.state_dict()
    if q_model is not None:
        ckpt["q_model"] = q_model.state_dict()
    if dqn_loss is not None:
        ckpt["target_params"] = dqn_loss.target_value_network_params.state_dict()
    if epsilon is not None:
        ckpt["epsilon"] = epsilon

    torch.save(ckpt, path)
    if args.wandb:
        wandb.save(path)
    # logging.debug(f"Saved checkpoint phase_000.pth (epoch={epoch}, iter={iteration})")


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.wandb:
        with open(args.wandb_api_path) as f:
            os.environ['WANDB_API_KEY'] = f.read().strip()
        wandb.login()
        wandb.init(project=args.project, config=vars(args), resume=False)

    if not check_cuda_availability() and not torch.backends.mps.is_available():
        logging.error("CUDA unavailable.")
        sys.exit(1)
    mp.set_start_method('spawn', force=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    base_loss = BetaBernoulliLoss(reduction=None)
    loss_fn = BalancedPUQuantileLoss(
        pi_real=args.alpha_prior / (args.alpha_prior + args.beta_prior),
        base_loss=base_loss
    ).to(device)
    train_metrics = FineTuningMetrics(
        thresholds=[i / 10 for i in range(2, 8)], device=device,
        alpha_prior=args.alpha_prior, beta_prior=args.beta_prior
    )
    val_metrics = FineTuningMetrics(
        thresholds=[i / 10 for i in range(2, 8)], device=device,
        alpha_prior=args.alpha_prior, beta_prior=args.beta_prior
    )

    # --- model & optimizer ---
    if args.load_latest_checkpoint:
        (
            input_emb, model, ref_emb, classifier, optimiser, q_model, dqn_loss,
            value_head, epsilon, start_epoch, _
        ) = \
            load_latest_checkpoint(args, device)
    else:
        if args.pre_trained_path:
            input_emb, model = load_pretrained_model(args, device)
            start_epoch = 0
        else:
            input_emb, model = instantiate_model(args, device)
            start_epoch = 0

        ref_emb = None if args.no_reference else NucleotideEmbeddingLayer(args.emb_dim, mlm_mode=True).to(device)

        classifier = BetaDistributionClassifier(
            input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2,
            using_reference_embedding=not args.no_reference
        ).to(device)

        # min_lr = args.base_lr / 10 if args.pre_trained_path else args.base_lr / 3
        min_lr = args.base_lr / 3
        pgs = get_layerwise_param_groups(model, args.base_lr, min_lr)
        pgs.append({'params': list(input_emb.parameters()), 'lr': min_lr})

        if args.use_RL:
            q_model, value_head, dqn_loss = instantiate_rl_model(
                args, device, input_emb, model, classifier, ref_emb
            )
        else:
            q_model = value_head = dqn_loss = None

        clsps = list(classifier.parameters()) + \
                ([] if args.no_reference else list(ref_emb.parameters())) + \
                ([] if not args.use_RL else list(value_head.parameters()))
        pgs.append({'params': clsps, 'lr': args.base_lr})
        optimiser = AdamW(pgs, eps=1e-9, weight_decay=0.01)

    if args.use_RL:
        target_updater = SoftUpdate(
            dqn_loss,
            eps=0.99
        )

        rb_pos, rb_neg = instantiate_replay_buffers(args)

        # start with epsilon being less greedy and more random
        epsilon = 1.0

    params_to_clip = list(chain(
        model.parameters(),
        classifier.parameters(),
        input_emb.parameters(),
        ref_emb.parameters() if ref_emb is not None else [],
        value_head.parameters() if args.use_RL else []
    ))

    # --- data loader once ---
    val_loader = create_finetuning_dataloader(
        csv_path=f"{args.finetune_metadata_dir}/test_fold_{args.fold}.csv",
        artefact_bam_path=args.artefact_bam_path,
        mutation_bam_path=args.mutation_bam_path,
        batch_size=200,
        base_quality_pad_idx=input_emb.base_quality_embeddings.padding_idx,
        cigar_pad_idx=input_emb.cigar_embeddings.padding_idx,
        is_first_pad_idx=input_emb.mate_pair_embeddings.padding_idx,
        mapped_to_reverse_pad_idx=input_emb.strand_embeddings.padding_idx,
        position_pad_idx=-1,
        max_read_length=args.max_read_length,
        shuffle=True,
        num_workers=4,
        prefetch_factor=1,
        balanced=True
    )
    vb = next(iter(val_loader))
    try:
        val_loader._shutdown_workers()
    except Exception:
        pass
    del val_loader
    gc.collect()
    val_batch = {
        k: (vb[k].to(device) if isinstance(vb[k], torch.Tensor) else vb[k])
        for k in vb.keys()
    }
    if 'mut_pos' in val_batch:
        val_batch['mut_pos'] = val_batch['mut_pos'].unsqueeze(-1)
    del vb
    if args.use_RL:
        ns = val_batch['nucleotide_sequences'].to(device)
        bq = val_batch['base_qualities'].to(device)
        ce = val_batch['cigar_encoding'].to(device)
        isf = val_batch['is_first'].to(device)
        m2r = val_batch['mapped_to_reverse'].to(device)
        pos = val_batch['positions'].to(device)
        rs = val_batch['read_support'].to(device)
        val_lbl = val_batch['labels'].to(device)
        mutp = val_batch['mut_pos'].to(device)
        ref = val_batch.get('reference', None)
        if ref is not None:
            ref = ref.to(device)

        idx = torch.nonzero(pos == mutp, as_tuple=True)

        if idx[0].shape[0] != ns.size(0):
            keep = set(idx[0].tolist())
            batch_idx = torch.arange(ns.size(0), device=device)
            mask = torch.tensor([i in keep for i in batch_idx], device=device)
            ns = ns[mask]
            bq = bq[mask]
            ce = ce[mask]
            isf = isf[mask]
            m2r = m2r[mask]
            pos = pos[mask]
            rs = rs[mask]
            val_lbl = val_lbl[mask]
            mutp = mutp[mask]
            if ref is not None:
                ref = ref[mask]

        val_obs = TensorDict({
            "nucleotide_sequences": ns,
            "base_qualities": bq,
            "cigar_encoding": ce,
            "is_first": isf,
            "mapped_to_reverse": m2r,
            "positions": pos,
            "mut_pos": mutp,
            **({"reference": ref}
               if "reference" in val_batch else {})
        }, batch_size=[val_batch["labels"].shape[0]]).to(device)
        del ns, bq, ce, isf, m2r, pos, rs, mutp, ref

        if not args.pre_trained_path:
            del val_batch

    dataset = create_finetuning_dataloader(
        csv_path=f"{args.finetune_metadata_dir}/train_fold_{args.fold}.csv",
        artefact_bam_path=args.artefact_bam_path,
        mutation_bam_path=args.mutation_bam_path,
        batch_size=args.batch_size,
        base_quality_pad_idx=input_emb.base_quality_embeddings.padding_idx,
        cigar_pad_idx=input_emb.cigar_embeddings.padding_idx,
        is_first_pad_idx=input_emb.mate_pair_embeddings.padding_idx,
        mapped_to_reverse_pad_idx=input_emb.strand_embeddings.padding_idx,
        position_pad_idx=-1,
        max_read_length=args.max_read_length,
        shuffle=True,
        num_workers=min(get_allocated_cpus() - 1, 8),
        prefetch_factor=1,
        # balanced=False
        balanced=True
    )

    epoch_iters = len(dataset)
    logging.info(f"Iterations in an epoch: {epoch_iters}")
    total_iters = (epoch_iters * max(args.epochs, 1))

    # wait until 1/10th of the first epoch has passed before saving
    # save_threshold = epoch_iters // 10

    max_lr_list = [g['lr'] for g in optimiser.param_groups]
    # mut_weight = get_effective_number(torch.tensor(2500.0, device=device))

    if args.pre_trained_path:
        phase = 1
    else:
        phase = 2

    iters = 0
    best_val_loss = float('inf')
    # best_val_weighted_pr_auc = 0.0
    window_size = 50
    pr_window = deque(maxlen=window_size)
    best_window_mean = -float('inf')

    data_iter = iter(dataset)
    scheduler = None

    subphase_idx = 0
    # total_groups = len(optimiser.param_groups) - 1

    while phase <= 2:
        # restart iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            batch = next(data_iter)

        # phase-specific setup on first iter of phase
        if iters == 0:
            # last_vals, last_prs, last_brier = [], [], []
            trigger = 0
            if phase == 1:
                logging.info(">>> Phase 1: Burn‑in <<<")
                # freeze all but classifier
                for i, g in enumerate(optimiser.param_groups):
                    for p in g['params']:
                        p.requires_grad = (i == len(optimiser.param_groups) - 1)
                # scheduler for burn_in_iters
                total_steps = args.burn_in_iters
                scheduler = OneCycleLR(
                    optimiser, max_lr=args.base_lr * 2,
                    total_steps=total_steps,
                    pct_start=0.0, anneal_strategy='cos',
                    cycle_momentum=False,
                    div_factor=1000.0
                )

            elif phase == 2:
                logging.info(">>> Phase 2: Main fine‑tune <<<")
                optimiser = AdamW(
                    optimiser.param_groups,
                    eps=1e-9, weight_decay=0.01
                )
                # unfreeze all if no pre-trained model
                if not args.pre_trained_path:
                    for g in optimiser.param_groups:
                        for p in g['params']:
                            p.requires_grad = True

                    total_steps = total_iters
                else:
                    steps_left_over = total_iters - iters_used_by_burnin
                    # total_steps = steps_left_over // (args.phases_per_epoch + 1)
                    total_steps = steps_left_over
                    # unfreeze all of the layers indicated by the phase
                    # if the value is 1 then unfreeze the top layer (index 0)
                    layers_to_unfreeze = args.phases_per_epoch
                    unfreeze_layers_by_epoch(
                        optimiser.param_groups,
                        epoch=layers_to_unfreeze,
                        ignore_groups=[len(optimiser.param_groups) - 1],
                    )

                # total_steps = 200
                if not args.use_RL:
                    scheduler = OneCycleLR(
                        optimiser, max_lr=max_lr_list,
                        total_steps=total_steps,
                        pct_start=0.3 if args.pre_trained_path else 0.0,
                        anneal_strategy='cos',
                        cycle_momentum=False,
                        div_factor=25.0,
                        final_div_factor=100.0
                    )
                else:
                    # set the OneCycleLR scheduler so that it is a flat
                    # non-decaying scheduler for the RL phase
                    scheduler = OneCycleLR(
                        optimiser, max_lr=max_lr_list,
                        total_steps=total_steps,
                        pct_start=0.0 if not args.pre_trained_path else 0.3,
                        anneal_strategy='cos',
                        cycle_momentum=False,
                        div_factor=1.0 if not args.pre_trained_path else 25.0,
                        final_div_factor=1.0
                    )
                # calculate the greedy epsilon decay constant
                if args.use_RL:
                    # decay constant for epsilon greedy so that it decays to
                    # 0.01 over half the total steps
                    decay_constant = np.exp(np.log(0.1 / epsilon) / (total_iters / 4))

            logging.info(f"Total steps for this phase: {total_steps}")
            iters = 0

        # # After 1000 iterations
        # if phase == 2 and iters == 1000 and loss_fn.top_k == 0:
        #     # Update the top_k value to 5
        #     loss_fn.top_k = 5

        ###############
        # forward/back
        ###############
        input_emb.train()
        model.train()
        classifier.train()
        if ref_emb is not None:
            ref_emb.train()

        ns = batch['nucleotide_sequences'].to(device)
        bq = batch['base_qualities'].to(device)
        ce = batch['cigar_encoding'].to(device)
        isf = batch['is_first'].to(device)
        m2r = batch['mapped_to_reverse'].to(device)
        pos = batch['positions'].to(device)
        rs = batch['read_support'].to(device)
        lbl = batch['labels'].to(device)
        ref = batch['reference'].to(device) if ref_emb is not None else None
        mutpos = batch['mut_pos'].to(device).unsqueeze(-1)

        idx = torch.nonzero(pos == mutpos, as_tuple=True)

        if idx[0].shape[0] != ns.size(0):
            keep = set(idx[0].tolist())
            batch_idx = torch.arange(ns.size(0), device=device)
            mask = torch.tensor([i in keep for i in batch_idx], device=device)
            ns = ns[mask]
            bq = bq[mask]
            ce = ce[mask]
            isf = isf[mask]
            m2r = m2r[mask]
            pos = pos[mask]
            rs = rs[mask]
            lbl = lbl[mask]
            mutpos = mutpos[mask]
            if ref is not None:
                ref = ref[mask]

        del batch

        if args.use_RL:
            current_obs = TensorDict(
                {
                    "nucleotide_sequences": ns.detach(),
                    "base_qualities": bq.detach(),
                    "cigar_encoding": ce.detach(),
                    "is_first": isf.detach(),
                    "mapped_to_reverse": m2r.detach(),
                    "positions": pos.detach(),
                    "mut_pos": mutpos.detach(),
                    # If you have a reference embedding, include it here:
                    **({"reference": ref.detach()} if ref_emb is not None else {})
                },
                batch_size=[ns.shape[0]],
            )
            next_obs = current_obs.clone()  # next state is the same in this case

        with (device_context(device)):

            if args.use_RL:
                q_model.train()
                # alphas, betas, V = q_model(obs_td)  # forward pass to compute Q-values
                # q_vals = alpha_beta_to_q(alphas, betas, V)  # (B, 2)

                q_vals = q_model(current_obs)  # (B, 2)

                # epsilon greedy action selection
                greedy = q_vals.argmax(dim=1)
                rand = torch.randint_like(greedy, high=2)
                action_int = torch.where(
                    torch.rand_like(greedy, dtype=torch.float32) < epsilon,
                    rand, greedy
                )
                action_onehot = torch.nn.functional.one_hot(
                    action_int, num_classes=2
                ).to(
                    torch.float32
                )  # shape [batch,2]

                raw_reward = torch.zeros_like(lbl)
                # recreate the reward structure from Wang et al. 2023
                # -\eta + 1 or -1 # for correct / incorrect predictions
                # where eta is randomly sampeld between 0 and 0.1.
                # mask = action_int != lbl  # only incorrect ones
                #
                # diff = torch.abs(action_int - lbl)  # shape (B,)
                # mx = torch.maximum(action_int, lbl) # shape (B,)
                # term = torch.zeros_like(diff, dtype=torch.float).to(device)

                # fill in only the incorrect entries
                # term[mask] = diff[mask].float() / mx[mask].float()
                eta1 = torch.rand(1, dtype=torch.float32, device=device) * 0.1
                eta2 = torch.rand(1, dtype=torch.float32, device=device) * 0.1

                # reward = torch.rand_like(lbl, dtype=torch.float32) * -0.5
                # raw_reward[(action_int == 1) & (lbl == 1)] += 10  # true positive
                # raw_reward[(action_int == 0) & (lbl == 1)] += -5  # false negative
                # raw_reward[(action_int == 0) & (lbl == 0)] += 0.1  # true negative
                # raw_reward[(action_int == 1) & (lbl == 0)] += -100  # false positive

                raw_reward[(action_int == 1) & (lbl == 1)] += 1  # true positive
                raw_reward[(action_int == 0) & (lbl == 1)] += -1  # false negative
                raw_reward[(action_int == 0) & (lbl == 0)] += 1  # true negative
                raw_reward[(action_int == 1) & (lbl == 0)] += -20  # false positive.  20 seems to be the best value so far.

                reward = raw_reward

                # # slight intra-class asymmetry
                # reward[lbl == 1] -= eta1
                # reward[lbl == 0] -= eta2

                # correct actions
                correct = (action_int == lbl)
                reward[correct] -= eta1

                # incorrect
                reward[~correct] -= eta2

                # subtract the mean reward to center it around 0
                reward = reward - reward.mean()

                # Divide by batch standard deviation so we have unit variance
                reward = reward / (reward.std() + 1e-9)

                reward = reward.unsqueeze(-1)

                transition = TensorDict({
                    "observation": current_obs,
                    ("next", "observation"): next_obs,
                    "action": action_onehot,
                    ("next", "reward"): reward,
                    ("next", "done"): torch.zeros_like(reward, dtype=torch.bool).detach(),
                    ("next", "terminated"): torch.zeros_like(reward, dtype=torch.bool, device=device)
                },
                    batch_size=[ns.shape[0]]
                )

                pos_mask = (lbl == 1)  # positives (True) / negatives (False)
                neg_mask = ~pos_mask

                if pos_mask.any():
                    rb_pos.extend(transition[pos_mask])
                if neg_mask.any():
                    rb_neg.extend(transition[neg_mask])

                batch_pos = rb_pos.sample().to(device)  # (B/2, …)
                batch_neg = rb_neg.sample().to(device)  # (B/2, …)

                # cat along batch-dim (dim=0) – td_cat keeps tensordict structure intact
                batch_td = torch.cat([batch_pos, batch_neg], 0)

                td_out = dqn_loss(batch_td)
                loss = td_out["loss"]

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params_to_clip, max_norm=1.0
                )
                optimiser.step()
                target_updater.step()

                if phase == 2:
                    # epsilon decay
                    epsilon = max(
                        0.1, epsilon * decay_constant
                    )

                # # get pt for metrics
                # pt = (q_vals[:, 1] - q_vals[:, 0])  # (B,)
            else:
                inp = input_emb(ns, ce, bq, m2r, isf)
                out = model(inp, pos)
                refemb = ref_emb(ref).squeeze(-2) if ref_emb is not None else None
                idx = torch.nonzero(pos == mutpos, as_tuple=True)
                cin = out[idx]
                alphas, betas = classifier(cin, refemb)
                alphas, betas = alphas.squeeze(-1), betas.squeeze(-1)

                rdw = get_effective_number(rs)

                # pt = alphas / (alphas + betas + 1e-8)
                lw = 1.0 / rdw

                pt = alphas / (alphas + betas + 1e-8)

                loss = loss_fn(alphas, betas, lbl, lw)

                preds = (pt > 0.5).float()
                tp = (preds * lbl).sum()
                tn = ((1 - preds) * (1 - lbl)).sum()
                fp = (preds * (1 - lbl)).sum()
                fn = ((1 - preds) * lbl).sum()
                tpr = tp / (tp + fn) if tp + fn > 0 else 0
                tnr = tn / (tn + fp) if tn + fp > 0 else 0
                balanced_acc = (tpr + tnr) / 2

                pr_with_prior = pr_auc_at_prior(
                    lbl.detach().to(torch.int32),
                    pt.detach(),
                    pi=args.alpha_prior / args.beta_prior
                )

                optimiser.zero_grad()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params_to_clip, max_norm=1.0
            )

            with torch.no_grad():  # no autograd bookkeeping
                grad_sq_sum = torch.tensor(0.0, device=device)
                # n_grads = 0
                for p in optimiser.param_groups[-1]['params'] + \
                         [p for g in optimiser.param_groups[:-1] for p in g['params']]:
                    if p.grad is not None and p.requires_grad:
                        grad_sq_sum += p.grad.pow(2).sum()
                        # n_grads += 1
                global_grad_norm = torch.sqrt(grad_sq_sum).item()

            optimiser.step()
            if phase == 1:
                # burn-in phase
                if iters < args.burn_in_iters - 1:
                    scheduler.step()
            else:
                # normal phase
                if iters < total_steps - 1:
                    scheduler.step()

        # validation
        if phase == 1 or not args.use_RL:
            val_loss, valm, dpos, dneg = run_validation(
                args, model, input_emb, classifier,
                val_batch, loss_fn, device, val_metrics,
                ref_base_embedding=ref_emb
            )
        else:
            valm = run_validation_rl(
                args, q_model, val_obs, val_lbl, device
            )
            val_loss = 0.0
        val_metrics.reset()

        pr_window.append(valm['PR_AUC_with_prior'])
        window_mean = sum(pr_window) / len(pr_window)

        # Get gradient norm values

        # debug metric printing
        logging.debug(
            f"[Phase {phase} | Iter {iters}] "
            f"LR={max(scheduler.get_last_lr()):.6f} Loss={loss:.6f} "
            f"ValLoss={val_loss:.6f}"
            # f"DiffMut={dpos:.6f} DiffArt={dneg:.6f}
        )
        if not args.use_RL:
            logging.debug(
                f"DiffMut={dpos:.6f} DiffArt={dneg:.6f} "
            )
            # tabulate train thresholds
            tdata = []
            train_metrics.update(
                alphas.detach(), betas.detach(), lbl.detach().to(torch.int32)
            )
            tdict = train_metrics.compute()
            train_metrics.reset()
            # for th in train_metrics.thresholds:
            #     tdata.append(
            #         [
            #             th,
            #             tdict[f'Precision@{th}'],
            #             tdict[f'Recall@{th}'],
            #             tdict[f'F1-Score@{th}']
            #         ]
            #     )
            # logging.debug("\n" + tabulate(tdata, headers=["T", "Prec", "Rec", "F1"], floatfmt=".4f"))
            # overall metrics
            logging.debug(
                f"Train ROC AUC={tdict['ROC AUC']:.4f} PR AUC={tdict['PR AUC']:.4f} "
                f"Brier={tdict['Brier Score']:.4f} "
                f"(with prior: {tdict['Brier Score (With prior)']:.4f}) "
                f"ECE={tdict['Calibration Error (ECE)']:.4f} "
                f"(with prior: {tdict['Calibration Error (With prior)']:.4f})"
            )
            logging.debug(
                f"Train Balanced Accuracy={balanced_acc:.4f} "
                f"(TPR={tpr:.4f} TNR={tnr:.4f}) "
                f"PR AUC (with prior)={pr_with_prior:.4f}"
            )

        # val
        logging.debug(
            f"Val ROC AUC={valm['ROC AUC']:.4f} PR AUC={valm['PR AUC']:.4f} "
            f"Brier={valm['Brier Score']:.4f} "
            f"(with prior: {valm['Brier Score (With prior)']:.4f}) "
            f"ECE={valm['Calibration Error (ECE)']:.4f} "
            # f"(with prior: {valm['Calibration Error (With prior)']:.4f})"
        )
        logging.debug(
            f"Val Balanced Accuracy={valm['Balanced Accuracy']:.4f} "
            f"(TPR={valm['TPR']:.4f} TNR={valm['TNR']:.4f}) \n"
            f"Val G-Mean (Harmonic mean of sensitivity and specicifity)={valm['G-Mean']:.4f} \n"
            f"Val F-score (Harmonic mean of precision and recall)={valm['F-Score']:.4f} \n"
            f"FPR={valm['FPR']:.4f} FNR={valm['FNR']:.4f} \n"
            f"PR AUC (with prior)={valm['PR_AUC_with_prior']:.4f} \n"
            f"Val NDCG={valm['NDCG']:.4f} \n"
            f"Val Overlap Coefficient (Balanced)={valm['Overlap Coefficient (Balanced)']:.4f} \n"
            f"Val Bayes Error={valm['Bayes Error']:.4f} \n"
            f"Val weighted precision@k: k=1: {valm['Precision@1']}, "
            f"k=2: {valm['Precision@2']}, "
            f"k=5: {valm['Precision@5']}, "
            f"k=10: {valm['Precision@10']}, "
            f"k=20: {valm['Precision@20']}, "
            f"k=30: {valm['Precision@30']}, "
            f"k=50: {valm['Precision@50']}"

        )
        if args.use_RL:
            logging.debug(
                f"Gradient Norm: {global_grad_norm:.4f} Epsilon: {epsilon} \n"
            )

        if args.wandb:
            if not args.use_RL:
                log = {
                    # train
                    "iter_loss": loss.item(),
                    "iter_lr": max(scheduler.get_last_lr()),
                    "train_ROC AUC": tdict['ROC AUC'],
                    "train_PR AUC": tdict['PR AUC'],
                    "train_PR AUC (with prior)": pr_with_prior,
                    "train_Brier": tdict['Brier Score'],
                    "train_Brier (with prior)": tdict['Brier Score (With prior)'],
                    "train_ECE": tdict['Calibration Error (ECE)'],
                    "train_ECE (with prior)": tdict['Calibration Error (With prior)'],
                    # validation
                    "val_loss": val_loss,
                    "val_ROC AUC": valm['ROC AUC'],
                    "val_PR AUC": valm['PR AUC'],
                    "val_PR AUC (with prior)": valm['PR_AUC_with_prior'],
                    "val_Brier": valm['Brier Score'],
                    "val_Brier (with prior)": valm['Brier Score (With prior)'],
                    "val_ECE": valm['Calibration Error (ECE)'],
                    # "val_ECE (with prior)": valm['Calibration Error (With prior)'],
                    # 1st strand 2nd strand probability difference
                    "diff_mut": dpos,
                    "diff_art": dneg,
                    # balanced accuracy
                    "train_balanced_accuracy": balanced_acc,
                    "train_TPR": tpr,
                    "train_TNR": tnr,
                    "val_balanced_accuracy": valm['Balanced Accuracy'],
                    "val_TPR": valm['TPR'],
                    "val_TNR": valm['TNR'],
                    "val_FPR": valm['FPR'],
                    "val_FNR": valm['FNR'],
                    "val_G-Mean": valm['G-Mean'],
                    "val_F-Score": valm['F-Score'],
                    "val_NDCG": valm['NDCG'],
                    "val_overlap_coef": valm['Overlap Coefficient (Balanced)'],
                    # "val_Bayes_error": valm['Bayes Error'],
                    "gradient_norm": global_grad_norm,
                    "val_precision@1": valm['Precision@1'],
                    "val_precision@2": valm['Precision@2'],
                    "val_precision@5": valm['Precision@5'],
                    "val_precision@10": valm['Precision@10'],
                    "val_precision@20": valm['Precision@20'],
                    "val_precision@30": valm['Precision@30'],
                    "val_precision@50": valm['Precision@50'],
                }
            else:
                log = {
                    # train
                    "iter_loss": loss.item(),
                    "iter_lr": max(scheduler.get_last_lr()),
                    # validation
                    "val_loss": val_loss,
                    "val_ROC AUC": valm['ROC AUC'],
                    "val_PR AUC": valm['PR AUC'],
                    "val_PR AUC (with prior)": valm['PR_AUC_with_prior'],
                    "val_Brier": valm['Brier Score'],
                    "val_Brier (with prior)": valm['Brier Score (With prior)'],
                    "val_ECE": valm['Calibration Error (ECE)'],
                    # 1st strand 2nd strand probability difference
                    "diff_mut": dpos,
                    "diff_art": dneg,
                    # balanced accuracy
                    "train_balanced_accuracy": balanced_acc,
                    "train_TPR": tpr,
                    "train_TNR": tnr,
                    "val_balanced_accuracy": valm['Balanced Accuracy'],
                    "val_TPR": valm['TPR'],
                    "val_TNR": valm['TNR'],
                    "val_FPR": valm['FPR'],
                    "val_FNR": valm['FNR'],
                    "val_G-Mean": valm['G-Mean'],
                    "val_NDCG": valm['NDCG'],
                    "val_overlap_coef": valm['Overlap Coefficient (Balanced)'],
                    # "val_Bayes_error": valm['Bayes Error'],
                    # gradient norm
                    "gradient_norm": global_grad_norm,
                    "val_precision@1": valm['Precision@1'],
                    "val_precision@2": valm['Precision@2'],
                    "val_precision@5": valm['Precision@5'],
                    "val_precision@10": valm['Precision@10'],
                    "val_precision@20": valm['Precision@20'],
                    "val_precision@30": valm['Precision@30'],
                    "val_precision@50": valm['Precision@50'],
                    "epsilon": epsilon,
                }
            wandb.log(
                log
            )

        # if iters >= save_threshold:
        if not args.use_RL and best_val_loss > val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(
                args, input_emb, model, classifier, optimiser, phase,
                iters, ref_emb
            )

            # if window_mean > best_window_mean:
            #     best_window_mean = window_mean
            #     save_best_checkpoint(
            #         args, input_emb, model, classifier, optimiser, phase,
            #         iters, ref_emb, "best_mean_val_PR_AUC"
            #     )

        if iters % 10000 == 0 and iters > 0:
            logging.info(
                f"Checkpointing at iteration {iters}..."
            )
            save_best_checkpoint(
                args, input_emb, model, classifier, optimiser, phase,
                iters, ref_emb, f"checkpoint_{iters:07d}",
                value_head=value_head if args.use_RL else None,
                q_model=q_model if args.use_RL else None,
                epsilon=epsilon if args.use_RL else None,
                dqn_loss=dqn_loss if args.use_RL else None
            )

        if iters >= total_steps - 1:

            if phase == 1:
                phase = 2
                iters_used_by_burnin = iters
            elif phase == 2:
                save_best_checkpoint(
                    args, input_emb, model, classifier,
                    optimiser, phase, iters, ref_emb,
                    "final" if not args.use_RL else None,
                    value_head=value_head if args.use_RL else None,
                    q_model=q_model if args.use_RL else None,
                    epsilon=epsilon if args.use_RL else None,
                    dqn_loss=dqn_loss if args.use_RL else None
                )
                phase = 3

            iters = 0
            continue

        iters += 1

    logging.info("All phases complete.")
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
