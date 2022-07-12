from modules.ioueval import *
import torch
import torch.nn.functional as F
from common.avgmeter import *
import yaml
losses = AverageMeter()
acc = AverageMeter()
iou = AverageMeter()

weight = np.random.rand(7) # float64
weight = torch.from_numpy(weight)

output = torch.rand(2, 7, 64, 2048).cuda()
label = torch.rand(2, 64, 2048).long().cuda()
config_dir = 'config/labels/semantic-kitti-7class.yaml'
config = yaml.safe_load(open(config_dir, 'r'))
train_squeeze = config["learning_map_inv"]
print(len(train_squeeze))

print("done")




