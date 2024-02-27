from toolkit import clock
import os  
from PIL import Image
import random
from torchvision import transforms,datasets 
import time

task = "dataset1"
inpath = "./Image/动漫人物"

# 一般划分数据集的方法
def set_generate(inpath,taskname):
    save_root = os.path.join("./Dataset",taskname)
    os.makedirs(save_root,exist_ok=True)
    os.makedirs(os.path.join(save_root,"train"),exist_ok=True)
    os.makedirs(os.path.join(save_root,"test"),exist_ok=True)
    os.makedirs(os.path.join(save_root,"val"),exist_ok=True)
    if input("是否使用默认数据集划分标准？(y/n)") == "y":
        train_rate = 0.6
        test_rate = 0.2
        val_rate = 0.2
    else:
        print("请输入训练集的比例.......")
        train_rate = float(input(">>")) 
        print("请输入测试集的比例.......")
        test_rate = float(input(">>"))
        print("请输入验证集的比例.......")
        val_rate = float(input(">>"))

    all_class_sets = os.listdir(inpath)
    class_no = len(all_class_sets)

    print(f"共计{class_no}种类别的数据....")
    print(f"{clock()}：开始生成数据集....")
    start = time.time()
    for class_set in all_class_sets:
        sub_start = time.time()
        img_path =os.path.join(inpath,class_set)
        print(f"{clock()}: 开始生成类别<<{class_set}>>数据......")
        img_list = []  
        for root, dirs, files in os.walk(img_path):  
            for file in files:  
                if file.lower().endswith(('.jpg', '.jpeg')):  # 添加你需要的图片格式  
                    image_path = os.path.join(root, file)  
                    image = Image.open(image_path)  
                    img_list.append(image)  
        print(f"共计{len(img_list)}张图片")

        print("正在划分数据集......")
        train_set = random.sample(img_list,int(train_rate*len(img_list)))
        img_list = [img for img in img_list if img not in train_set]
        test_set = random.sample(img_list,int(test_rate*len(img_list)))
        img_list = [img for img in img_list if img not in test_set]
        val_set = random.sample(img_list,int(val_rate*len(img_list)))
        
        train_path = os.path.join(save_root,"train")
        test_path = os.path.join(save_root,"test")
        val_path = os.path.join(save_root,"val")

        print("正在生成训练集.......")
        img_save(os.path.join(train_path,class_set),train_set)
        
        print("正在生成测试集.......")
        img_save(os.path.join(test_path,class_set),test_set)

        print("正在生成验证集.......")
        img_save(os.path.join(val_path,class_set),val_set)

        print(f"{clock()}：<<{class_set}>>数据集划分完成！耗时{time.time()-sub_start}s")
    print(f"{clock()}：所有数据集划分完成！耗时{time.time()-start}s")


def img_save(save_path,img_list):
    if not os.path.exists(save_path):  
        os.makedirs(save_path)
    for index, image in enumerate(img_list):  
        new_filename = f'{index+1}.png'
        image.save(os.path.join(save_path, new_filename))   

set_generate(inpath,task)
    

