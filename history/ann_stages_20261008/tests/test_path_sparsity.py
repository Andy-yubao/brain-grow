"""仅新路径正则的一阶零保护/联动缩放及物理删除风险，单个CPU检查。"""
import copy

import torch

from braingrow.models import StructuredMLP
from braingrow.path_sparsity import path_group_penalty, path_node_scores, delete_hidden_nodes


def test_path_groups_rescaling_zero_gradient_and_physical_deletion():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = StructuredMLP([3, 4, 3, 2])
        x = torch.tensor([[.2, .5, .8], [.9, .1, .4]])
        original = model(x).detach()
        scores = [value.detach().clone() for value in path_node_scores(model)]
        scale = float(sum(value.sum() / width for value, width in zip(scores, model.widths[1:-1])))
        penalty = path_group_penalty(model, model.widths, scale)
        penalty.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in list(model.parameters())[:-1])  # 输出偏置不在隐藏路径中。
        transformed = copy.deepcopy(model)
        with torch.no_grad():
            for index, factors in enumerate((torch.tensor([.25, 2., 3., .5]), torch.tensor([4., .5, 2.]))):
                layer, following = transformed.layers[index:index+2]
                layer.theta.view(layer.out_features, layer.in_features).mul_(factors[:, None])
                layer.bias.mul_(factors)
                following.theta.view(following.out_features, following.in_features).div_(factors[None, :])
        torch.testing.assert_close(transformed(x), original, atol=1e-6, rtol=1e-5)
        for before, after in zip(scores, path_node_scores(transformed)):
            torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)
        compact = delete_hidden_nodes(model, 0, [1])
        assert compact.widths == (3, 3, 3, 2)
        torch.testing.assert_close(compact.layers[0].expanded_weight(), model.layers[0].expanded_weight()[[0, 2, 3]])
        torch.testing.assert_close(compact.layers[1].expanded_weight(), model.layers[1].expanded_weight()[:, [0, 2, 3]])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.zero_grad(set_to_none=True)
        zero_penalty = path_group_penalty(model, model.widths, scale)
        zero_penalty.backward()
        assert float(zero_penalty.detach()) == 0
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
