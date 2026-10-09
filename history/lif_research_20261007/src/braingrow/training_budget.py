"""下一轮SNN训练预算；供新数据入口接入，不启动训练或读取数据。"""
from dataclasses import dataclass
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "next_snn_training.json"
HARD_EPOCH_CAP = 25


@dataclass(frozen=True)
class TrainingBudget:
    maximum_epochs: int
    control_interval_epochs: int
    fit_interval_epochs: int

    def __post_init__(self):
        for value in (self.maximum_epochs, self.control_interval_epochs, self.fit_interval_epochs):
            if type(value) is not int or value < 1:
                raise ValueError("轮数及评价间隔须为正整数")
        if self.maximum_epochs > HARD_EPOCH_CAP:
            raise ValueError("下一轮总预算最多25个epoch")

    def remaining_epochs(self, completed_epochs, progress_epoch=None):
        if type(completed_epochs) is not int or not 0 <= completed_epochs <= self.maximum_epochs:
            raise ValueError("已完成轮数超出下一轮总预算；不能把历史轮数重置为0接续")
        if progress_epoch is not None and (
            type(progress_epoch) is not int
            or progress_epoch != completed_epochs + 1
            or progress_epoch > self.maximum_epochs
        ):
            raise ValueError("在途轮数不符或超过25轮总预算")
        # 在途轮属于尚未完成部分；恢复其未完成批次，不另加一轮。
        return self.maximum_epochs - completed_epochs

    def evaluation_epochs(self, pool):
        intervals = {"control": self.control_interval_epochs, "fit": self.fit_interval_epochs}
        interval = intervals[pool]
        return tuple(sorted(set(range(interval, self.maximum_epochs + 1, interval)) | {self.maximum_epochs}))


def load_budget(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["epoch_accounting"] != "total_including_resume" or config["evaluate_final_epoch"] is not True:
        raise ValueError("须累计恢复轮数并评价最后一轮")
    return TrainingBudget(*(config[key] for key in (
        "maximum_epochs", "control_interval_epochs", "fit_interval_epochs")))


if __name__ == "__main__":
    budget = load_budget()
    print(json.dumps(dict(maximum_epochs=budget.maximum_epochs,
        control_epochs=budget.evaluation_epochs("control"),
        fit_epochs=budget.evaluation_epochs("fit"),
        training_started=False), ensure_ascii=False, indent=2))
