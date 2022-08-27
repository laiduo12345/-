import cv2 
from PIL import Image
import numpy
import scipy.special
import matplotlib.pyplot

def convertPicture(path):
    img=Image.open(path)
    img_black=img.convert('L')
    img_black=img_black.resize((28,28))
    #img_black.show()
    binary_matrix=[]
    #遍历每一个像素
    for y in range(img_black.height):
     for x in range(img_black.width):
            v=255-img_black.getpixel((x,y))
            if v>120:
               binary_matrix.append(v)
            else:
               binary_matrix.append(0)
    return binary_matrix


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

#创建对象
n=neuralNetwork(784,100,10,0.3)
        
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
binary_matrix=convertPicture("pic.png")
print("\n\n\n\n\nanswer:"+str(numpy.argmax(n.query(binary_matrix))))
print(n.query(binary_matrix))
image_array=numpy.asfarray(binary_matrix).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation="None")
matplotlib.pyplot.show()



