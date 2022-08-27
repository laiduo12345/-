import numpy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
       #初始化节点个数
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #初始化权重
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate
        pass

    def query(self,inputs_list):
        #初始化
        inputs=numpy.array(inputs_list,ndmin=2).T
    
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=scipy.special.expit(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=scipy.special.expit(final_inputs)
        return final_outputs

    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=scipy.special.expit(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=scipy.special.expit(final_inputs)

        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)

        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),hidden_outputs.T)
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),inputs.T)
        pass

def draw(X,Y,Z):
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i],Y[i],Z[i],c="grey")
        ax.set_xlabel('epoch')
        ax.set_ylabel('angle')
        ax.set_zlabel('rate')
    tmp=numpy.argmax(Z)
    ax.scatter(X[tmp],Y[tmp],Z[tmp],c="red")
    matplotlib.pyplot.show()
#创建对象
epoch=[]
angle=[]
rate=[]
        
#读取数据
training_data_file=open("mnist_train.csv",mode="r")
training_data_list=training_data_file.readlines()
training_data_file.close()

test_data_file=open("mnist_test.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

for epochs in range(1,11):
    for a in range(21):
        n=neuralNetwork(784,190,10,0.14/float(epochs))
        for e in range(epochs):
            #进行训练
            for record in training_data_list:
                all_values=record.split(",")
                inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
                inputs_plus=scipy.ndimage.rotate(inputs.reshape(28,28),a,cval=0.01,reshape=False).reshape(784)
                inputs_minus=scipy.ndimage.rotate(inputs.reshape(28,28),-a,cval=0.01,reshape=False).reshape(784)
                targets=numpy.zeros(10)+0.01
                targets[int(all_values[0])]=0.99
                n.train(inputs,targets)
                n.train(inputs_plus,targets)
                n.train(inputs_minus,targets)

#测试并计算正确率


        cnt=0
        correct=0
        for i in test_data_list:
            all_values=i.split(",")
            target=int(all_values[0])
            anslist=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
            ans=numpy.argmax(anslist)
            if ans==target:
                correct+=1
            cnt+=1
        epoch.append(epochs)
        angle.append(a)
        rate.append(float(correct)/float(cnt))

f=open("out.txt",mode="w",encoding="utf-8")
for i in range(len(epoch)):
    f.write(str(epoch[i])+" ")
f.write("\n")
for i in range(len(angle)):
    f.write(str(angle[i])+" ")
f.write("\n") 
for i in range(len(rate)):
    f.write(str(rate[i])+" ")
f.close()

draw(epoch,angle,rate)

#画出数字图像
# image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
# matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation="None")
# print(all_values[0])
# matplotlib.pyplot.show()