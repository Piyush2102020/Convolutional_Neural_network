import torch
from torch import nn
import torchvision
from torchvision import datasets
import numpy
import mlxtend
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer
from torch.utils.data import DataLoader

Batch_size=32
#setup train data
train_data=datasets.FashionMNIST(root="data",
                                 train=True,#train data will be provided
                                 download=True,
                                 transform=ToTensor(),#how to tranform the data
                                 target_transform=None)

def accuracy_fn(y_true,y_pred):
    correct=torch.eq(y_true,y_pred).sum().item()
    acc=(correct/len(y_pred))*100
    return acc
test_data=datasets.FashionMNIST(root="data",train=False,
                                download=True,
                                transform=ToTensor(),target_transform=None)

class_names=train_data.classes
print(len(train_data),len(test_data))
torch.manual_seed(42)
fig=plt.figure(figsize=(9,9))

train_dataloader=DataLoader(dataset=train_data,batch_size=Batch_size,shuffle=True)
test_dataloader=DataLoader(dataset=test_data,batch_size=Batch_size,shuffle=True)

train_feature_batch,train_label_batch=next(iter(train_dataloader))
torch.manual_seed(42)
random_idx=torch.randint(0,len(train_feature_batch),size=[1]).item()
img,label=train_feature_batch[random_idx], train_label_batch[random_idx]


def print_tran_time(start:float,end:float):
    #print the time between start and end
    return end-start/60
lr=0.1



class CNN(nn.Module):
    def __init__(self,input,out,hidden):
        super().__init__()
        self.conv_block_1=nn.Sequential(nn.Conv2d(in_channels=input,out_channels=hidden,kernel_size=3,stride=1,padding=1)
                                        ,nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,stride=1,padding=1)
                                        ,nn.ReLU()
                                        ,nn.MaxPool2d(kernel_size=2))
        self.conv_block_2=nn.Sequential(nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=3,stride=1,padding=1)
                                        ,nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))
        self._class_layer=nn.Sequential(nn.Flatten()
                                        ,nn.Linear(in_features=hidden*7*7,out_features=out)
                                        )
    def forward(self,x):
        return self._class_layer(self.conv_block_2(self.conv_block_1(x)))
        
        
torch.manual_seed=42
model=CNN(input=1,out=10,hidden=10)
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
loss_fn=nn.CrossEntropyLoss()
epochs=1
train_time=default_timer()

#create the train loop
for epoch in tqdm(range(epochs)):
    print()
    print(f"Epoch : {epoch} \n-----")
    train_loss=0
    for batch,(X,y) in enumerate(train_dataloader):
        model.train()
        # print(X.shape) 32 1 28 28
        logits=model(X)
        loss=loss_fn(logits,y)
        train_loss+=loss #add loss for every batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch %400==0:
            print(f"looked at {batch*len(X)}/{len(train_dataloader.dataset)} samples.")
        #Divide total train loss by lenght of train dataloader
    train_loss/=len(train_dataloader)
    
    #testing
    test_loss,test_acc=0,0
    model.eval()
    with torch.inference_mode():
        for X,y in test_dataloader:
            test_pred=model(X)
            test_loss+=loss_fn(test_pred,y)
            test_acc+=accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim=1))
        #test loss
        test_loss/=len(test_dataloader)
        test_acc/=len(test_dataloader)
        print(f"\nTrain Loss : {train_loss:.4f} | Test loss: {test_loss:.4f} | Testt acc : {test_acc:.4f}")


train_time_end =default_timer()

print(print_tran_time(start=train_time,end=train_time_end))            
            
# conv_layer=nn.Conv2d(1,10,kernel_size=3)
# flatten_layer=nn.Flatten()
# pool_layer=nn.MaxPool2d(kernel_size=3)


# fig, axs = plt.subplots(2, 2)

# # print(img,label)

# axs[0,0].imshow(img.squeeze())
# axs[0,0].set_title("Original Image")
# axs[0,0].set_ylabel(f"Shape : {str(img.shape[0])}, {str(img.shape[1])}, {str(img.shape[2])}")

# print(img.shape)
# con_image=conv_layer(img)
# axs[0,1].imshow(con_image[0].detach().numpy())
# print(con_image.shape)
# axs[0,1].set_title("Convoluted Image")
# axs[0,1].set_ylabel(f"Shape : {str(con_image[0].shape[0])}, {str(con_image[0].shape[1])}")

# pool_image=pool_layer(con_image)
# print(pool_image[0].shape)
# axs[1,0].set_title("Pooling")
# axs[1,0].imshow(pool_image[0].detach().numpy())
# axs[1,0].set_ylabel(f"Shape : {str(pool_image[0].shape[0])}, {str(pool_image[0].shape[1])}")

# # plt.imshow(pool_image.detach().numpy())

# flatten_image=flatten_layer(pool_image)
# print(flatten_image.shape)
# axs[1,1].set_title("Flattened Image (Batch)")
# axs[1,1].imshow(flatten_image.detach().numpy())
# axs[1,1].set_ylabel(f"Shape : {str(flatten_image.shape[0])}, {str(flatten_image.shape[1])}")

# plt.show()


#train loop

def make_prediction(model:torch.nn.Module,data:list):
    pred_probs=[]
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample=sample.unsqueeze(dim=0)
            print(sample.shape)
            pred_logit=model(sample)
            pred_prob=torch.softmax(pred_logit.squeeze(),dim=0)
            pred_probs.append(pred_prob)
            
        return torch.stack(pred_probs)
        
import random
random.seed(42)
test_sample=[]
test_label=[]

for sample,label in random.sample(list(test_data),k=9):
    test_sample.append(sample)
    test_label.append(label)
    
    
# plt.imshow(test_sample[0])
# plt.show()
pred_prob=make_prediction(model=model,data=test_sample)
pred_label=pred_prob.argmax(dim=0)

nrows=3
ncols=3

for i ,sample in enumerate(test_sample):
    plt.subplot(nrows,ncols,i+1)
    plt.imshow(sample.squeeze())
    # plt.title(class_names[pred_label[i]])
    truth_label=class_names[test_label[i]]
    title_test=f"Pred : {class_names[pred_label[i]]} | Truth {truth_label} "
    if class_names[pred_label[i]] ==  truth_label:
        plt.title(title_test,c="g",fontsize=7)
        
    else:
        plt.title(title_test,c="g",fontsize=7)
        
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
plt.show()