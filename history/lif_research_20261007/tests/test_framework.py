"""保留框架的行为检查：脉冲通信、输出累积、结构编辑与保存恢复。"""
import numpy as np
import pytest
from braingrow.network.growing import GrowingNetwork
from braingrow.data.moving_mnist_v5 import render
def digit():
    a = np.zeros((28,28), np.uint8)
    a[4:24,6:10] = 255
    a[18:24,10:23] = 180
    a[6:9,11:15] = 120
    return a


def test_accumulator_has_no_leak_fire_or_bonus_and_resets():
    net = GrowingNetwork.create(n=1)
    net.win[:]=0; net.input_mask[:]=False
    net.rec[:]=0; net.rec_mask[:]=False
    net.out[:]=0; net.output_mask[:]=False
    net.out[0,2]=2; net.output_mask[0,2]=True
    net.out[0,14]=3; net.output_mask[0,14]=True
    net.bias[:]=2; net.alpha[:]=0
    x=np.zeros((2,8,784),np.float32)
    p,a,t=net.forward(x,True)
    assert p[0,2]==64 and p[0,14]==96
    assert a["output_spikes"]==0 and a["hidden_spikes"]==64
    assert a["mean_spikes_per_image"]==4 and a["mean_spikes_per_video"]==32
    assert p[0,:10].argmax()==2 and p[0,10:].argmax()==4
    np.testing.assert_array_equal(p,net.forward(x)[0])
    # 前两帧发放后完全静默，末态仍保留积分。
    net.bias[:]=0; net.win[0,0]=2; net.input_mask[0,0]=True
    x[:,:2,0]=1
    p,_,_=net.forward(x)
    assert p[0,2]==16 and p[0,14]==24


def test_split_preserves_function_and_output_categories(tmp_path):
    net=GrowingNetwork.create(n=8)
    x=np.random.default_rng(11).uniform(0,.7,(3,8,784)).astype(np.float32)
    before=net.forward(x)[0]
    twin=net.clone(); twin.split_node(3)
    assert twin.valid() and twin.n==9
    np.testing.assert_allclose(before,twin.forward(x)[0],atol=1e-5)
    assert twin.out.shape[1]==17
    twin.delete_node(8); assert twin.valid() and twin.n==8
    with pytest.raises(ValueError): twin.delete_node(17)
    path=tmp_path/"model.npz"; net.save(path)
    loaded=GrowingNetwork.load(path)
    np.testing.assert_array_equal(before,loaded.forward(x)[0])
    assert loaded.max_nodes is None and loaded.max_edges is None
    loaded.max_nodes=7; assert not loaded.valid()
    loaded.max_nodes=None; loaded.allocation_guard_bytes=1
    with pytest.raises(MemoryError): loaded.valid()


def test_membrane_without_spikes_cannot_communicate():
    net=GrowingNetwork.create(n=2)
    net.win[:]=0; net.input_mask[:]=False
    net.rec[:]=0; net.rec_mask[:]=False
    net.out[:]=0; net.output_mask[:]=False
    net.bias[:]=0; net.alpha[:]=0
    net.bias[0]=.9
    net.rec[0,1]=100; net.rec_mask[0,1]=True
    net.out[1,0]=1; net.output_mask[1,0]=True
    p,a,_=net.forward(np.zeros((1,8,784),np.float32))
    assert a["hidden_spikes"]==0 and not p.any()


def test_augmented_poses_independent_direction_and_variant():
    x,m=render(digit(),11,0)
    xx,mm=render(digit(),11,0)
    np.testing.assert_array_equal(x,xx)
    assert x.shape==(7,8,784)
    poses=np.asarray(m["poses_xy_clockwise_degrees"])
    assert np.isclose(poses[5,-1,2]-poses[5,0,2],105)
    assert np.isclose(poses[6,-1,2]-poses[6,0,2],-105)
    assert not np.array_equal(x,render(digit(),11,1)[0])
    assert not np.array_equal(x[5,::-1],x[6])
    with pytest.raises(ValueError): render(digit(),11,seed=12)



def test_data_geometry_and_pixel_reference():
    from braingrow.data.rendering import foreground, render_pose, pose_bounds
    x,m = render(digit(),11)
    tile,bbox = foreground(digit())
    poses = np.asarray(m["poses_xy_clockwise_degrees"])
    bounds = pose_bounds(bbox,m["scale"],poses)
    assert bounds[...,:2].min() >= 1 and bounds[...,2:].max() <= 27
    for pixels,pose in zip(x.reshape(-1,28,28),poses.reshape(-1,3),strict=True):
        reference = render_pose(tile,m["scale"],pose,padding=8)
        center=reference[8:36,8:36]
        np.testing.assert_array_equal(center,np.rint(pixels*255).astype(np.uint8))
        assert center.any() and reference.sum()==center.sum()


def test_direct_potential_metrics():
    from braingrow.evaluation.metrics import evaluate
    p=np.zeros((2,17));p[0,2]=5;p[0,14]=4;p[1,3]=5;p[1,15]=4
    result=evaluate(p,np.array([[2,4],[3,6]]))
    assert result["digit_accuracy"]==1 and result["motion_accuracy"]==.5
    assert result["joint_accuracy"]==.5 and result["motion_confusion"][6][5]==1


def test_id_split_and_cache_fail_closed(tmp_path):
    from braingrow.data.dynamic import split_ids, prepare
    from braingrow.data.moving_mnist_v5 import prepare as prepare_v5
    labels=np.repeat(np.arange(10),40)
    splits=split_ids(labels)
    assert len(splits["train"])==200 and len(splits["validation"])==100
    sets=[set(v) for v in splits.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert splits==split_ids(labels)
    with pytest.raises(ValueError): prepare(tmp_path,seed=12)
    (tmp_path/"manifest_v5.json").write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError): prepare_v5(tmp_path)
