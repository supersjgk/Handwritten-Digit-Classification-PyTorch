import torch
import torch.nn as nn
import torchvision #for computer vision datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=28*28
hidden_size=100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.001

#MNIST handwriting dataset
train_data=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

test_data=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

#printing a sample
ex=next(iter(train_loader))
sample,label=ex
#print(sample.shape, label.shape, sample, label)

for i in range(20):
	plt.subplot(5,4,i+1)
	plt.imshow(sample[i][0],cmap='gray')
plt.show()

class NN(nn.Module):
	def __init__(self,inp_size,hid_size,num_classes):
		super(NN,self).__init__()
		self.linear1=nn.Linear(inp_size,hid_size)
		self.relu=nn.ReLU()
		self.linear2=nn.Linear(hid_size,num_classes)
	
	def forward(self,x):
		out=self.linear1(x)
		out=self.relu(out)
		out=self.linear2(out)
		return out

model=NN(input_size,hidden_size,num_classes)

loss=nn.CrossEntropyLoss() #applies softmax itself at the end layer
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training
num_steps=len(train_loader)
for epoch in range(num_epochs):
	for i,(image,label) in enumerate(train_loader):
		image=image.reshape(-1,28*28).to(device)
		label=label.to(device)

		#forward pass
		output=model(image)
		l=loss(output,label)

		#backward pass
		optimizer.zero_grad()
		l.backward()

		#update parameters
		optimizer.step()

		if (i+1)%100==0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_steps}, loss={l.item():.4f}')

#testing
with torch.no_grad():
	n_correct,n_samples=0,0
	for image,label in test_loader:
		image=image.reshape(-1,28*28).to(device)
		label=label.to(device)
		output=model(image)

		_,pred=torch.max(output,1) #to get class labels with max probability, returns value,index
		n_samples+=label.shape[0]
		n_correct+=(pred==label).sum().item()
	accuracy=100.0*n_correct/n_samples
	print(f'accuracy={accuracy}')
