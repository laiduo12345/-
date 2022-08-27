import numpy
import scipy.special
import matplotlib.pyplot
from PIL import Image
import scipy.special
import scipy.ndimage

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
        #输入层
        inputs=numpy.array(inputs_list,ndmin=2).T
        #隐藏层
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=scipy.special.expit(hidden_inputs)
        #输出层
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=scipy.special.expit(final_inputs)
        #返回结果
        return final_outputs

    def train(self,inputs_list,targets_list):
        #输入层
        inputs=numpy.array(inputs_list,ndmin=2).T               
        targets=numpy.array(targets_list,ndmin=2).T
        #隐藏层
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=scipy.special.expit(hidden_inputs)
        #输出层
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=scipy.special.expit(final_inputs)
        #反向传播误差
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #梯度下降
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),hidden_outputs.T)
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),inputs.T)
        pass

    def output(self):
        file=open("data.txt",mode="w",encoding="utf-8")
        for i in range(len(self.wih)):
            for j in range(len(self.wih[i])):
                file.write(str(self.wih[i][j])+" ")
            file.write("\n")
        for i in range(len(self.who)):
            for j in range(len(self.who[i])):
                file.write(str(self.who[i][j])+" ")
            file.write("\n")
        file.close()
        pass

    def input(self):
        file=open("data.txt",mode="r",encoding="utf-8")
        file_data=file.readlines()
        file.close()

        for i in range(len(self.wih)):
            line_data=file_data[i].split(" ")
            for j in range(len(self.wih[i])):
                self.wih[i][j]=line_data[j]
        for i in range(len(self.who)):
            line_data=file_data[i+len(self.wih)].split(" ")
            for j in range(len(self.who[i])):
                self.who[i][j]=line_data[j]
        pass


#画出数字图像
def draw_num(all_values):
    image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation="None")
    print(all_values[0])
    matplotlib.pyplot.show()

#转换图片
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

def predict(path):
    binary_matrix=convertPicture(path)
    n=neuralNetwork(784,190,10,0.014)
    n.input()
    anslist=n.query((numpy.asfarray(binary_matrix)/255.0*0.99)+0.01)
    ans=numpy.argmax(anslist)
    return ans

#创建对象
# n=neuralNetwork(784,190,10,0.014)
# n.input()
#读取数据
# training_data_file=open("mnist_train.csv",mode="r")
# training_data_list=training_data_file.readlines()
# training_data_file.close()

#进行训练
# for e in range(10):
#     for record in training_data_list:
#         all_values=record.split(",")
#         inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
#         inputs_plus=scipy.ndimage.rotate(inputs.reshape(28,28),9,cval=0.01,reshape=False).reshape(784)
#         inputs_minus=scipy.ndimage.rotate(inputs.reshape(28,28),-9,cval=0.01,reshape=False).reshape(784)
#         targets=numpy.zeros(10)+0.01
#         targets[int(all_values[0])]=0.99
#         n.train(inputs,targets)
#         n.train(inputs_plus,targets)
#         n.train(inputs_minus,targets)

#测试并计算正确率
# test_data_file=open("mnist_test.csv","r")
# test_data_list=test_data_file.readlines()
# test_data_file.close()

# cnt=0
# correct=0
# for i in test_data_list:
#     all_values=i.split(",")
#     target=int(all_values[0])
#     anslist=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
#     ans=numpy.argmax(anslist)
#     if ans==target:
#         correct+=1
#     cnt+=1
# print(float(correct)/float(cnt))
# n.output()
# draw_num(test_data_list[0].split(","))

# ans=predict("pic1.png")

