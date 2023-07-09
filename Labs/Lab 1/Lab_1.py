import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

df = pd.read_csv('pima-indians-diabetes.csv')

pregs = df['pregs'].tolist()
plas = df['plas'].tolist()
pres = df['pres'].tolist() 
skin = df['skin'].tolist()
test = df['test'].tolist()
bmi = df['BMI'].tolist()
pedi = df['pedi'].tolist()
age = df['Age'].tolist()
cls = df['class'].tolist()

attributes = ["pregs","plas","pres","skin","test","bmi","pedi","age","cls"]
attributes_lst = [pregs,plas,pres,skin,test,bmi,pedi,age,cls]

#Question 1

def question1(list,str):
    print("The mean for " + str + " column is = ",np.mean(list))
    print("The median for " + str + " column is = ",np.median(list))
    print("The mode for " + str + " column is = ",st.mode(list))
    print("The maximum for " + str + " column is = ",max(list))
    print("The minimum for " + str + " column is = ",min(list))

for i in range (len(attributes)):
    if (i!=8):
        question1(attributes_lst[i],attributes[i])

print("")
print("---------------------------------------------------------------------------------------------")

#Question 2

def scat_age(list,str):
    plt.scatter(age,list,edgecolors="red")
    plt.xlabel("Age")
    plt.ylabel(str)
    arg = str + " v/s age graph"
    plt.title(arg)
    plt.show()

for i in range (len(attributes)):
    if (i==7 or i==8):
        continue
    else:
        print("The scatter plot for the "+ attributes[i] +" col w.r.t the age is:")
        scat_age(attributes_lst[i],attributes[i])

def scat_bmi(list,str):
    plt.scatter(bmi,list,color='red',edgecolors="black")
    plt.xlabel("BMI")
    plt.ylabel(str)
    arg = str + " v/s BMI graph"
    plt.title(arg)
    plt.show()

for i in range (len(attributes)):
    if (i==5 or i==8):
        continue
    else:
        print("The scatter plot for the "+ attributes[i] +" col w.r.t the BMI is:")
        scat_bmi(attributes_lst[i],attributes[i])

print("")
print("---------------------------------------------------------------------------------------------")

#Question 3

def corr_age(list):
    print(np.corrcoef(age, list)[0, 1])

for i in range (len(attributes)):
    if(i==7 or i==8):
        continue
    else:
        print("The correlation of "+attributes[i]+" column with age is:",end = " ")
        corr_age(attributes_lst[i])
    

print("")
print("")

def corr_bmi(list):
    print(np.corrcoef(bmi, list)[0, 1])

for i in range (len(attributes)):
    if(i==5 or i==8):
        continue
    else:
        print("The correlation of "+attributes[i]+" column with BMI is:",end = " ")
        corr_bmi(attributes_lst[i])
print("")
print("---------------------------------------------------------------------------------------------")

#Question 4

plt.hist(pregs,edgecolor = "red")
plt.xlabel('Number of times pregnant')  
plt.ylabel('Number of times')  
plt.show()

plt.hist(skin,color = 'red',edgecolor = "black")
plt.xlabel('Tricep fold thickness(mm)')
plt.ylabel('Number of times')
plt.show()

print("")
print("---------------------------------------------------------------------------------------------")

#Question 5

df1 = df.groupby(["class"], as_index=False)
preg_1 = (df1.get_group(1)["pregs"]).tolist()
preg_0 = (df1.get_group(0)["pregs"]).tolist()
plt.hist(preg_0,edgecolor = "red")
plt.xlabel('Number of times pregnant')
plt.title('Class = 0')
plt.show()

plt.hist(preg_1,color = 'red',edgecolor = "black")
plt.xlabel('Number of times pregnant')
plt.title('Class = 1')
plt.show()

print("")
print("---------------------------------------------------------------------------------------------")

#Question 6

def box(list,str):
    plt.title("Boxplot of "+str)
    plt.boxplot(list, patch_artist = True, notch ='True', vert = 0)
    plt.grid()
    plt.show()

for i in range (len(attributes)):
    if (i!=8):
        box(attributes_lst[i],attributes[i])

print("")
print("---------------------------------------------------------------------------------------------")