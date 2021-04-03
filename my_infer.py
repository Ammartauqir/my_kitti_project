import subprocess
import datetime
import yaml
import os
import shutil
from my_infer.user import *

dataset = "/media/ammar/HDD/LIDAR_datasets/KITTISem/dataset/"
log = '/logs/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/'
model = 'model_load_dir/squeezesegV2'

try:
    ARCH = yaml.safe_load(open(model + "/arch_cfg.yaml", 'r'))
    DATA = yaml.safe_load(open(model + "/data_cfg.yaml", 'r'))
except Exception as e:
    print(e)
    print("Error opening arch or data yaml file.")
    quit()

# create log folder
if os.path.isdir(log):
    shutil.rmtree(log)
    os.makedirs(log)
    os.makedirs(os.path.join(log, "sequences"))
for seq in DATA["split"]["train"]:
    seq = '{0:02d}'.format(int(seq))
    print("train", seq)
    os.makedirs(os.path.join(log, "sequences", seq))
    os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
for seq in DATA["split"]["valid"]:
    seq = '{0:02d}'.format(int(seq))
    print("valid", seq)
    os.makedirs(os.path.join(log, "sequences", seq))
    os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
for seq in DATA["split"]["test"]:
    seq = '{0:02d}'.format(int(seq))
    print("test", seq)
    os.makedirs(os.path.join(log, "sequences", seq))
    os.makedirs(os.path.join(log, "sequences", seq, "predictions"))

if os.path.isdir(model):
    print("model folder exists! Using model from %s" % (model))
else:
    print("model folder doesnt exist! Can't infer...")
    quit()

  # create user and infer dataset
user = User(ARCH, DATA, dataset, log, model)
user.infer()