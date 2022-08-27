import numpy
import scipy.special
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

per={}

for a in numpy.arange(0.1,0.91,0.1):
    #创建对象
    n=neuralNetwork(784,100,10,a)

#读取数据
    training_data_file=open("mnist_train.csv",mode="r")
    training_data_list=training_data_file.readlines()
    training_data_file.close()

#进行训练
    for record in training_data_list:
        all_values=record.split(",")
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(10)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

#测试并计算正确率
    test_data_file=open("mnist_test.csv","r")
    test_data_list=test_data_file.readlines()
    test_data_file.close()

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
    per[a]=float(correct)/float(cnt)
print(per)






#画出数字图像
# image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
# matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation="None")
# print(all_values[0])
# matplotlib.pyplot.show()