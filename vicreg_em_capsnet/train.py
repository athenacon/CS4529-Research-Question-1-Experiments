from pathlib import Path
import argparse 
import os
import random 
import neptune
from torch import nn, optim
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from utilscapsnet import AverageMeter
import resnet
from capsnet import resnet20 
# ref https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/7
seed_value = 42
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)
 

class ApplyTransform(Dataset):
    # reference:https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if transform is None and target_transform is None:
            print("Transforms have failed")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)

def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "vicreg_trained_on_imgnet" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
           
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=350,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    # run = neptune.init_run(project = 'enter your username and project file', dependencies="infer",
    # api_token="enter your api token")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _ = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load("resnet50.pth", map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=True)

    head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to(device)
    
    model = nn.Sequential(backbone, head).to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    
    for param in head.parameters():
        param.requires_grad = True
          
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    start_epoch = 0
 
    from torchvision import datasets
    from torch.utils.data import random_split 
      
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
     
    full_training_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=None)
    
    train_transform = transforms.Compose(
    [   
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    )
    
    test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Data loading code
    training_dataset_size = 45000
    val_dataset_size = 5000
    
    train_dataset, val_dataset = random_split(full_training_dataset, [training_dataset_size, val_dataset_size])
    
    train_dataset = ApplyTransform(train_dataset, transform=train_transform)
    val_dataset = ApplyTransform(val_dataset, transform=test_transform)
    
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)

    
    best_valid_acc = 0
    for epoch in range(start_epoch, args.epochs):
         
        backbone.eval()
        head.train()
        losses = AverageMeter()
        accs = AverageMeter()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device) 
            out = model(x)
            
            loss = criterion(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # compute gradients and update SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss, train_acc = losses.avg, accs.avg
        # run["after_pretraining/training/epoch/loss"].log(train_loss)
        # run["after_pretraining/training/epoch/acc"].log(train_acc)
        # evaluate on validation set
        with torch.no_grad():
            # valid_loss, valid_acc = validate(epoch)
            
            model.eval()

            losses = AverageMeter()
            accs = AverageMeter()

            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                out = model(x)
                
                loss = criterion(out, y)

                # compute accuracy
                pred = torch.max(out, 1)[1]
                correct = (pred == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])
        
        valid_loss, valid_acc = losses.avg, accs.avg
        
        run["after_pretraining/validation/epoch/loss"].log(valid_loss)
        run["after_pretraining/validation/epoch/acc"].log(valid_acc)  
        
        # decay lr
        scheduler.step()
    save_checkpoint(
    {'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'valid_acc': valid_acc
        } 
    )
    # Testing
    correct = 0
    model.eval()
    num_test = len(test_loader.dataset)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test, perc, error)
    )
    
    # run["after_pretraining/testing/loss"].log(error)
    # run["after_pretraining/testing/acc"].log(perc)

    # run.stop() 
        
if __name__ == "__main__":
    main()