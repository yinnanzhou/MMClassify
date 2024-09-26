import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from MMClassifyFunc.train import Trainer
from MMClassifyFunc.models import CustomResNet
from MMClassifyFunc.data_preprocess import get_loader
from MMClassifyFunc.data_read import get_data
from MMClassifyFunc.visualization import visualize_results

folder_path = r'/home/mambauser/MMClassify/data/dataPng/continuous_0_300'
in_channels = 3

samples, labels = get_data(
    folder_path=folder_path,
    in_channels=in_channels,
    # wordIndex=list(range(30)),
    # fileIndex=list(range(0,10))+list(range(12,30))+list(range(32,40)),
    # fileIndex=list(range(0,10))+list(range(30,40)),
    # fileIndex=list(range(0,40)),
    personIndex=list(range(15)),
    # txIndex=[0,1,2,3,4],
)

print("len(samples): {}".format(len(samples)))
print("len(set(labels)): {}".format(len(set(labels))))

trainloader, testloader = get_loader(samples=samples, labels=labels)

# classifier
classifier = CustomResNet(in_channels=in_channels,
                          num_classes=len(set(labels)),
                          weights=models.ResNet18_Weights.DEFAULT,
                        #   weights=None,
                          model='resnet18')

# optimizers
lr = 1e-3
betas = (.5, .99)
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)
criterion = nn.CrossEntropyLoss()

# train model
NUM_INPUTS = 1
epochs = 30

trainer = Trainer(
    num_inputs=NUM_INPUTS,
    classifier=classifier,
    optimizer=optimizer,
    criterion=criterion,
    print_every=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_cuda=torch.cuda.is_available(),
    use_scheduler=False)

trainer.train(trainloader=trainloader, testloader=testloader, epochs=epochs)

visualize_results(trainer=trainer)

torch.save(classifier.state_dict(), 'model.pth')