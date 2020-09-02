import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import os
import torch.nn.functional as F

transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class BenderDataset(Dataset):
    def __init__(self, characters, root_dir, num_per, transform=None):
        self.characters = characters
        self.root_dir = root_dir
        self.transform = transform
        self.num_per = num_per
        cwd = os.getcwd()
        target= os.path.join(cwd, root_dir)
        os.chdir(target)

    def __len__(self):
        return len(self.characters)*self.num_per

    def __getitem__(self, idx):
        character = self.characters[idx%len(self.characters)]
        index = idx//len(self.characters)
        if(self.root_dir == "test"):
            index += 16 # Very hackey
        os.chdir(character)
        #print(os.getcwd())
        im = Image.open(character + str(index) + ".jpg")
        os.chdir("..")
        im = self.transform(im)
        return (im, idx%len(self.characters))


class Net(torch.nn.Module):
    def __init__(self, num_characters):
        torch.nn.Module.__init__(self)
        self.conv1 = torch.nn.Conv2d(3, 16, 7, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)

        self.w1  = torch.nn.Linear(16*16*32, 30)
        self.w2 = torch.nn.Linear(30, num_characters)
        self.max = torch.nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.max(F.relu(self.conv1(x)))
        x = self.max(F.relu(self.conv2(x)))
        x = x.view(-1, 16*16*32)
        x = F.relu(self.w1(x))
        x = self.w2(x)
        return(x)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    characters = ["Aang", "Katara", "Toph", "Zuko"]
    bd = BenderDataset(characters, "train", 16, transform)
    dl = DataLoader(bd, batch_size = 4, shuffle = True)

    atlaNet = Net(4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(atlaNet.parameters())

    for epoch in range(5):
        dataIter = iter(dl)
        i = 0
        for img, label in dataIter:
            optimizer.zero_grad()
            out = atlaNet(img)
            loss = criterion(out, label)
            print("Loss: " + str(loss))
            loss.backward()
            optimizer.step()
            i+=1


    os.chdir("..")
    with torch.no_grad():
        testSet = BenderDataset(characters, "test", 4, transform)
        testLoader = DataLoader(testSet, batch_size = 1, shuffle = True)
        # Test performance on all test data
        correct = 0
        total = 0
        testIter = iter(testLoader)
        for img, label in testIter:
            out = atlaNet(img)
            if out.argmax() == label:
                correct += 1
            if total%4 == 0:
                print("I think this is .... " + characters[out.argmax()])
                print("It was really: " + characters[label])
                imshow(img.squeeze())
            total += 1
        print(str(correct) + " of " + str(total))

        # Test individual case
        # img, label = testSet[11]
        # plt.imshow(img.squeeze()); plt.show()
        # out = atlaNet(img.unsqueeze(0))
        # print(out)
        # print("I think this is .... " + characters[out.argmax()])
        # if out.argmax() == label:
        #     print("Yay!")
        # else:
        #     print("Ooops")

    # Now save the model
    PATH = '../avatar_path.pth'
    torch.save(atlaNet.state_dict(), PATH)
    # Can be loaded elsewhere with:
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

if __name__ == "__main__":
    main()
