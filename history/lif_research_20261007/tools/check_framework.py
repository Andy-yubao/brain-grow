"""实际v5输入通过保留框架；只做前向诊断，不训练。"""
import json
from braingrow.data.dynamic import prepare
from braingrow.network.growing import GrowingNetwork
from braingrow.evaluation.metrics import evaluate


def main():
    data=prepare()
    net=GrowingNetwork.create(seed=11,n=24)
    assert net.valid()
    x,y=data["train_x"][:7],data["train_y"][:7]
    p,activity,_=net.forward(x)
    assert p.shape==(7,17)
    print(json.dumps({"ok":True,"training_performed":False,"scope":"untrained_7_video_io_smoke_only",
        "input_shape":list(x.shape),"output_shape":list(p.shape),"resources":net.resources(),
        "activity":activity,"diagnostic_metrics":evaluate(p,y)},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
