from resnet import *
from torch.autograd import Variable
import struct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import DataLoader, sampler
from sklearn.metrics import accuracy_score




num_epoch = 100
num_classes = 10

plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['image.cmap'] = 'gray'

def show_images(images): #定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(images[i].reshape(sqrtimg, sqrtimg))
    return


labels_path = 'G:/Study/Deep_Learning_cs231n/MNIST/train-labels.idx1-ubyte'
images_path = 'G:/Study/Deep_Learning_cs231n/MNIST/train-images.idx3-ubyte'

with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)
with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), -1)


sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
images = images.reshape(images.shape[0], sqrtimg, sqrtimg)
images = images[:, np.newaxis, :, :]
images = images.astype(np.float32)
print(images.dtype)
images_zip = list(zip(images, labels))

images_test = images[0: 128]
test_images_label = labels[0: 128]
dataloader = DataLoader(dataset=images_zip, batch_size=128, shuffle=True)



net = ResNet18().cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

num = 0
for epoch in range(num_epoch):
    for img, label in dataloader:
        label = label.numpy()
        one_hot = np.zeros((label.shape[0], num_classes))
        one_hot[np.arange(label.shape[0]), label] = 1
        one_hot = one_hot.astype(np.float32)

        label = Variable(torch.from_numpy(one_hot)).float().cuda()
        z = Variable(img).float().cuda()
        g_label = net(z)
        loss = criterion(g_label, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if num % 50 == 0:
            z = Variable(torch.from_numpy(images_test)).float().cuda()
            test_label = net(z).data.cpu().numpy()
            test_label_2 = np.zeros(test_label.shape[0])
            for i in range(test_label.shape[0]):
                test_label_2[i] = np.argmax(test_label[i])
            acc = accuracy_score(test_images_label, test_label_2)
            print('Epoch[{}/{}], loss:{:.6f}, accuracy:{:.6f}'.format(epoch, num_epoch, loss, acc))
        num += 1





















