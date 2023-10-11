import torch
from einops import einsum
from monai.data import DataLoader
from torch.nn import NLLLoss

from dann import DANNModule, DANNTrainer
from dataset import AMOSDataset
from lib.loss.target_adaptive_loss import TargetAdaptiveLoss
from segmentation import SegmentationModule, SegmentationTrainer

# Settings
device = "cuda"
nc = 4
fg = [2]
bg = [1, 3]
root = r"C:\Users\User\Desktop\TSM_Project\data\amos22"

# Psuedo data
logit = torch.rand(2, nc, 3, 3, 3).cuda()
y = torch.argmax(logit, dim=1).cuda()
y[~torch.isin(y, torch.tensor(fg).cuda())] = 0

# Module
seg_mod = SegmentationModule().to(device)
dann_mod = DANNModule(ct_foreground=fg, mr_foreground=bg).to(device)
train_ds = AMOSDataset(root_dir=root, modality="ct", spatial_dim=3, stage="train", dev=True)
train_dtl = DataLoader(train_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
val_ds = AMOSDataset(root_dir=root, modality="ct", spatial_dim=3, stage="validation", dev=True)
val_dtl = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

batch = next(iter(train_dtl))
image, mask = batch["image"].to(device), batch["label"].to(device)

# Loss
loss = TargetAdaptiveLoss(num_class=nc, foreground=fg)

# Track NaN
# i81 = torch.load("./debug/images81.pth").cuda()
# m81 = torch.load("./debug/masks81.pth").cuda()
# i82 = torch.load("./debug/images82.pth").cuda()
# m82 = torch.load("./debug/masks82.pth").cuda()
# i83 = torch.load("./debug/images83.pth").cuda()
# m83 = torch.load("./debug/masks83.pth").cuda()
# i84 = torch.load("./debug/images84.pth").cuda()
# m84 = torch.load("./debug/masks84.pth").cuda()


# module = SegmentationModule(criterion=loss)
# module.load_state_dict(torch.load("./debug/seg_module80.ckpt"))
# module = module.cuda()

# module.update(i81, m81)
# module.update(i82, m82)
# module.update(i83, m83)

# module.optimizer.zero_grad()
# o83 = module(i83)
# loss_t = loss(o83, m83)
# loss_t.retain_grad()
# loss_t.backward()
# grad = [p.grad for p in module.parameters()]
