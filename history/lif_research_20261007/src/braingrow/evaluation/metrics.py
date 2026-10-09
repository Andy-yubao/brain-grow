"""直接比较两组输出电位；仅计算指标，不拟合额外分类器。"""
import numpy as np

def task_loss(p, y):
    total = 0.
    for start,stop,col in ((0,10,0),(10,17,1)):
        a = p[:,start:stop].astype(np.float64)
        a -= a.max(axis=1,keepdims=True)
        total += float((np.log(np.exp(a).sum(axis=1))-a[np.arange(len(y)),y[:,col]]).mean())
    return total


def evaluate(p, y):
    pred = np.column_stack((p[:,:10].argmax(axis=1),p[:,10:].argmax(axis=1)))
    cm=[]
    for col,k in ((0,10),(1,7)):
        table=np.zeros((k,k),int)
        np.add.at(table,(y[:,col],pred[:,col]),1)
        cm.append(table.tolist())
    return {"loss":task_loss(p,y),"digit_accuracy":float((pred[:,0]==y[:,0]).mean()),
            "motion_accuracy":float((pred[:,1]==y[:,1]).mean()),
            "joint_accuracy":float((pred==y).all(axis=1).mean()),
            "digit_confusion":cm[0],"motion_confusion":cm[1]}
