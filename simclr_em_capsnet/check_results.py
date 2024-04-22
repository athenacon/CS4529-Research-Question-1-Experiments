import argparse
import torch 
import torchvision.transforms as transforms
import numpy as np
from simclr import SimCLR
from simclr.modules import get_resnet 
from utils import yaml_config_hook
from torchvision import transforms

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
 
# Capsule Network
from capsule_network import resnet20
from simclr.modules.identity import Identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
        
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import datasets
    
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
  
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
        transform=test_transform)
    
   
    kwargs = dict(
        batch_size=32,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)

    capsule_network = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to(args.device)

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    encoder.avgpool = Identity() 
    encoder.fc = Identity()
    
    # Load the checkpoint
    checkpoint = torch.load("saved_checkpoint_for_evaluation_of_results/simclr_linear_evaluation_after_pretrained_ckpt_epoch_350.pth.tar")

    # initialize model
    model = SimCLR(encoder, capsule_network)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(args.device)
    print("Model loaded from pre-trained model successfully")
     
    # Testing
    correct = 0
    model.eval()
    num_test = len(test_loader.dataset)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(args.device), y.to(args.device)

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
    