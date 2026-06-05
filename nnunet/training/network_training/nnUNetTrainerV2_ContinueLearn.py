"""
nnUNetTrainerV2_ContinueLearn -- Continual Learning Trainer
============================================================
Used for Task026_ContinueLearn_Fold0:
  Fine-tunes from the Task020_Mouse5mod (5-mod, fold 0) checkpoint on new data,
  while suppressing catastrophic forgetting of previously learned knowledge.

Training commands:
  nnUNet_train 3d_fullres nnUNetTrainerV2_ContinueLearn 26 0 -p nnUNetPlansv2.1
  nnUNet_train 2d         nnUNetTrainerV2_ContinueLearn 26 0 -p nnUNetPlansv2.1

Strategy:
  1. Load weights from Task020 fold_0 model_best.model
  2. Use a lower initial learning rate (1e-4 vs. 1e-2) to protect old knowledge
  3. Optional EWC (Elastic Weight Consolidation) regularization (set USE_EWC=True)
"""

import os
import torch
import numpy as np
from torch.optim import SGD
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_ContinueLearn(nnUNetTrainerV2):
    """
    Continual learning trainer.
    - Automatically loads pretrained weights from Task020 fold_0
    - Reduces initial lr to 1e-4 (vs. default 1e-2)
    - Optional EWC regularization (set USE_EWC = True)
    """

    # ── Configuration ─────────────────────────────────────────────────────────
    # Source task for pretrained checkpoint
    PRETRAIN_TASK = "Task020_Mouse5mod"
    PRETRAIN_TRAINER = "nnUNetTrainerV2__nnUNetPlansv2.1"
    PRETRAIN_FOLD = 0

    # Continual learning hyperparameters
    INITIAL_LR = 1e-4          # 100x lower than default 1e-2 to protect old knowledge
    USE_EWC = False            # set True to enable EWC regularization
    EWC_LAMBDA = 1000.0        # EWC penalty coefficient
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, plans_file, fold, output_folder, dataset_directory,
                 batch_dice=True, stage=None, unpack_data=True,
                 deterministic=True, fp16=False):
        super().__init__(
            plans_file, fold, output_folder, dataset_directory,
            batch_dice, stage, unpack_data, deterministic, fp16,
        )
        self.initial_lr = self.INITIAL_LR
        self.ewc_fisher: dict = {}   # Fisher information matrix (EWC)
        self.ewc_params: dict = {}   # reference parameter snapshot (EWC)

    # ── Initialization: load pretrained weights ───────────────────────────────
    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        if training:
            self._load_pretrained_weights()

    def _locate_pretrained_checkpoint(self) -> str:
        """Locate the best checkpoint of Task020 fold_0 under RESULTS_FOLDER."""
        results_root = os.environ.get(
            "RESULTS_FOLDER",
            "/path/to/data/RESULTS_FOLDER",
        )
        ckpt = os.path.join(
            results_root, "nnUNet",
            # select 3d_fullres or 2d depending on current output folder
            "3d_fullres" if "3d" in self.output_folder else "2d",
            self.PRETRAIN_TASK,
            self.PRETRAIN_TRAINER,
            f"fold_{self.PRETRAIN_FOLD}",
            "model_best.model",
        )
        return ckpt

    def _load_pretrained_weights(self):
        ckpt_path = self._locate_pretrained_checkpoint()
        if not os.path.isfile(ckpt_path):
            self.print_to_log_file(
                f"[ContinueLearn] WARNING: pretrained checkpoint not found at\n"
                f"  {ckpt_path}\n"
                f"  Training from scratch instead."
            )
            return

        self.print_to_log_file(
            f"[ContinueLearn] Loading pretrained weights from:\n  {ckpt_path}"
        )
        saved_model = torch.load(ckpt_path, map_location=torch.device("cpu"))
        pretrained_dict = saved_model["state_dict"]

        model_dict = self.network.state_dict()
        # Only load weights whose shapes match (guards against num_classes mismatch)
        matched = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped = [k for k in pretrained_dict if k not in matched]
        model_dict.update(matched)
        self.network.load_state_dict(model_dict)

        self.print_to_log_file(
            f"[ContinueLearn] Loaded {len(matched)} / {len(pretrained_dict)} params. "
            f"Skipped: {skipped if skipped else 'none'}"
        )

        # EWC: save a reference parameter snapshot after loading pretrained weights
        if self.USE_EWC:
            self._snapshot_ewc_params()

    # ── Optimizer: use reduced lr ─────────────────────────────────────────────
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "initialize network first"
        self.optimizer = SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5,
        )
        self.lr_scheduler = None
        self.print_to_log_file(
            f"[ContinueLearn] SGD optimizer, initial_lr={self.initial_lr}"
        )

    # ── EWC ───────────────────────────────────────────────────────────────────
    def _snapshot_ewc_params(self):
        """Save a snapshot of the current (pretrained) parameters for EWC penalty."""
        self.ewc_params = {
            n: p.data.clone()
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        self.print_to_log_file("[ContinueLearn] EWC: parameter snapshot saved.")

    def compute_ewc_loss(self) -> torch.Tensor:
        """Elastic Weight Consolidation penalty term."""
        if not self.ewc_params:
            return torch.tensor(0.0, device=next(self.network.parameters()).device)
        loss = torch.tensor(0.0, device=next(self.network.parameters()).device)
        for n, p in self.network.named_parameters():
            if n in self.ewc_params:
                fisher = self.ewc_fisher.get(n, torch.ones_like(p.data))
                loss += (fisher * (p - self.ewc_params[n]) ** 2).sum()
        return self.EWC_LAMBDA * loss

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """Add EWC penalty on top of segmentation loss."""
        if not self.USE_EWC:
            return super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # Replicate parent-class forward pass, inserting EWC term before backward
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        output = self.network(data)
        del data

        seg_loss = self.loss(output, target)
        ewc_loss = self.compute_ewc_loss()
        total_loss = seg_loss + ewc_loss

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target

        if do_backprop:
            if self.fp16:
                with torch.cuda.amp.autocast():
                    pass  # already computed above
                self.amp_grad_scaler.scale(total_loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        return total_loss.detach().cpu().numpy()

    def print_to_log_file(self, *args, **kwargs):
        super().print_to_log_file(*args, **kwargs)


# ── Helper functions (mirrored from nnUNet internals) ─────────────────────────
def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, gpu_id=0):
    if isinstance(data, list):
        data = [to_cuda(i, gpu_id) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data
