"""ANN基线的控制损失反馈；按评价次数降率及早停，可完整恢复。"""
import math

import torch


POLICY = dict(metric="control.loss", mode="min", factor=0.5, patience=1,
              threshold=0.01, threshold_mode="rel", cooldown=0, min_lr=1e-5,
              eps=1e-8, early_stop_patience_checks=5, minimum_epochs=10,
              control_interval_epochs=2, fit_interval_epochs=5)


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **{key: POLICY[key] for key in
                      ("mode", "factor", "patience", "threshold", "threshold_mode",
                       "cooldown", "min_lr", "eps")})


def initial_feedback(scheduler):
    return dict(scheduler=scheduler.state_dict(), best_loss=None,
                stale_checks=0, last_control_epoch=0, reductions=0)


def step_feedback(scheduler, feedback, loss, epoch):
    loss = float(loss)
    if not math.isfinite(loss) or loss < 0 or epoch <= feedback["last_control_epoch"]:
        raise ValueError("控制损失须非负且有限，反馈轮数须严格递增")
    before = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    scheduler.step(loss)
    after = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    best = feedback["best_loss"]
    if best is None or loss < best * (1 - POLICY["threshold"]):
        feedback["best_loss"], feedback["stale_checks"] = loss, 0
    else:
        feedback["stale_checks"] += 1
    feedback["last_control_epoch"] = epoch
    feedback["reductions"] += int(before != after)
    feedback["scheduler"] = scheduler.state_dict()
    stop = (epoch >= POLICY["minimum_epochs"]
            and feedback["stale_checks"] >= POLICY["early_stop_patience_checks"])
    return dict(epoch=epoch, control_loss=loss, learning_rates_before=before,
                learning_rates_after=after, reduced=before != after,
                effective_epoch=epoch + 1, best_loss=feedback["best_loss"],
                stale_checks=feedback["stale_checks"], early_stop=stop)
