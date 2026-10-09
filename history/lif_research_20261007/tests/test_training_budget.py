"""实际下一轮配置的预算、恢复与末轮评价检查；不运行模型。"""
import pytest

from braingrow.training_budget import TrainingBudget, load_budget


def test_next_run_total_budget_and_final_evaluation():
    budget = load_budget()
    assert budget.maximum_epochs == 25
    assert budget.evaluation_epochs("control") == (*range(2, 25, 2), 25)
    assert budget.evaluation_epochs("fit") == (5, 10, 15, 20, 25)
    assert budget.remaining_epochs(0) == 25
    assert budget.remaining_epochs(24, progress_epoch=25) == 1
    assert budget.remaining_epochs(25) == 0


@pytest.mark.parametrize("completed,in_progress", [(25, 26), (26, None), (75, None), (24, 24)])
def test_resume_cannot_reset_or_exceed_budget(completed, in_progress):
    with pytest.raises(ValueError):
        load_budget().remaining_epochs(completed, in_progress)


def test_override_cannot_expand_to_26_epochs():
    with pytest.raises(ValueError, match="最多25"):
        TrainingBudget(26, 2, 5)
