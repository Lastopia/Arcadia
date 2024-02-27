import os
import sys
import json
import torch.nn as nn 
from PIL import Image
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from Models import DenseNet,GoogleNet,ResNet,VGG
import time
from toolkit import clock 

import torch
torch.cuda.current_device()

"""
迭代日志
1. 预测模型变成批量预测并自动统计预测查全率和查准率等数据
2. 训练模型数据做全，评价标准做全。
3. 统计运行时间数据
"""

# --------------------模型参数设定------------------------------- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        # 先在原图上随机采集不同宽高比和大小的子图片，再将子图片缩放为指定大小
        transforms.RandomHorizontalFlip(),
        # 以给定的概率随机水平旋转图像，默认为0.5
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        # 将张量归一化处理
    ]),

    "val": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

task_name = "task1"
model_name = "VGG"

root = "./Task/{}/{}".format(task_name,model_name)
batch_size = 1
epochs = 1
learn_rate = 0.0001
num_classes = 8

# --------------------训练数据设定--------------------------------- #
dataset = "dataset1"
data_path = "./Dataset/"+ dataset
train_path = os.path.join(data_path,"train")
val_path = os.path.join(data_path,"val")
test_path = os.path.join(data_path,"test")
net = VGG.Build(num_classes=num_classes)
model = net.to(device)
print(train_path)
def train():
    print("核心运行模式: {}".format(device))
    os.makedirs(root,exist_ok=True)
    train_dataset = datasets.ImageFolder(
        # datasets.ImageFolder()将一组图片加载到内存中，并为每个图像分配标签
        # 根目录下的文件夹名称为分类名称，每个分类文件夹中有图片,应该是输出字典吧
        # dataset.classes 返回["文件夹名称"]
        # dataset.class_to_idx 返回{"文件夹名称"：分类编号}
        # dataset.imgs 返回所有[("图片路径，分类编号")]
        root = train_path,
        transform = data_transform["train"]
    )

    class_dict = dict((val,key) for key,val in train_dataset.class_to_idx.items())
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
    print(class_dict)

    with open(os.path.join(root,"Classification.json"),'w') as json_file:
        json_file.write(json.dumps(class_dict,indent = 4))
    # indent 表示缩进距离

    nw = min([
        os.cpu_count(),
        # 返回CPU核心的数量
        batch_size if batch_size > 1 else 0, 8
    ])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = nw
    )
    validate_dataset = datasets.ImageFolder(
        root = val_path,
        transform = data_transform["val"]
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = nw
    )
    print("共计{}张图片用于训练,{}张图片用于测试".format(len(train_dataset), len(validate_dataset)))  

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = learn_rate)
    best_acc = 0.0
    train_steps = len(train_loader)
    train_loss = []
    val_accuracy = []
    print(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file = sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epcho [{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader,file = sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim = 1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()
        acc /= len(validate_dataset)
        val_accuracy.append(round(acc,3)) 
        train_loss.append(round(running_loss / train_steps,3))
        print("[epoch %d] Train loss: %.3f Val accuracy: %.3f" % (epoch + 1, train_loss[(epoch)], val_accuracy[(epoch)]))
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),os.path.join(root,"weights.pth"))
    
    fig,ax = plt.subplots()
    ax.plot(range(epochs),train_loss,color='blue',label='train loss',linestyle='-')
    ax.plot(range(epochs),val_accuracy,color='red',label='val accuracy',linestyle='-')
    ax.legend()
    plt.savefig(os.path.join(root,"Effect.png"))
    plt.show()

    
    print ("Finished Training!")

def predict():
    if os.path.exists(test_path):
        print("测试数据文件夹已经创建过了")
    else:
        os.makedirs(test_path)
        print("测试数据文件夹成功创建")

    img = Image.open(test_img)
    plt.imshow(img)
    img = data_transform['val'](img)
    img = torch.unsqueeze(img,dim = 0)
    with open (os.path.join(root,"Classification.json"),"r") as f:
        class_indict = json.load(f)
    model.load_state_dict(torch.load(os.path.join(root,"weights.pth"),map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim = 0)
        predict_cla = torch.argmax(predict).numpy()
    res = "class:{}  prob:{:.3f}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    plt.title(res)
    print(len(predict))
    for i in range(len(predict)):
        print("class:{:10}  prob:{:.3f}".format(class_indict[str(i)],predict[i].numpy()) )
        plt.show()

def main():
    print("当前使用模型为:-- {} --".format(model_name))
    print("当前执行任务为:-- {} --".format(task_name))
    operation = input("运行哪种模式？（0-train;1-predict）")
    if(operation == '0'):
        print("开始执行训练程序....")
        train()
    elif(operation == '1'):
        print("开始执行预测程序....")
        predict()
    else:
        print("指令错误，请重试！！")

if __name__ == "__main__":
    main()
