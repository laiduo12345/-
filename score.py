import neuralNetwork

def get_score(path1="-1",path2="-1",path3="-1"):
    ans1=ans2=ans3=0
    if path1 !="-1":
        ans1=neuralNetwork.predict(path1)
    if path2 !="-1":
        ans2=neuralNetwork.predict(path2)
    if path3 !="-1":
        ans3=neuralNetwork.predict(path3)
    score=ans1*100+ans2*10+ans3
    return score
    
path1="name/pic1.jpg"
path2="name/pic2.jpg"
path3="name/pic3.png"
print("score:"+str(get_score(path1,path2,path3)))