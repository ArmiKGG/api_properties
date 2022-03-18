import os
import sys
sys.path.append("../../monk_v1/")
sys.path.append("../../monk_v1/monk/")
from monk.gluon_prototype import prototype

gtf = prototype(verbose=1)
gtf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True)


def make_prediction(path):
    img_name = path
    predictions = gtf.Infer(img_name=img_name)
    return predictions


make_prediction('../test.jpg')

