import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#question 1

df = pd.read_csv("landslide_data3_miss.csv") #reading the csv database with missing data
orig_df = pd.read_csv("landslide_data3_original.csv") #reading the original csv
headers = list(df.columns.values) #getting all the headings as a list
use = df.isnull().sum() # getting the summ of all the null values in a column
plt.bar(headers,use[headers],width=0.6, color = "red" , edgecolor = "black") # plotting the data
plt.xticks(rotation=90)
plt.show()


#Question 2
#Part A
print("The initial number of rows = ", df.shape[0])
no_nan_df = df[df['stationid'].notna()]
print("The new number of rows = ", no_nan_df.shape[0])
print("Therefore the number of rows removed = ", (df.shape[0] - no_nan_df.shape[0]))
#Part B
print("The initial number of rows after part(a) = ", no_nan_df.shape[0])
new_df = no_nan_df[no_nan_df.isnull().sum(axis=1) < 3] #axis = 1 represents columns and axis = 0 represents rows
print("The new number of rows = ", new_df.shape[0])
print("Therefore the number of rows further removed = ", (no_nan_df.shape[0] - new_df.shape[0]))


#Question 3
print('Number of missing values after step 2:')
print(new_df.isnull().sum())
print('Total number of missing values from the new database=',new_df.isnull().sum().sum())


#Question 4
index=[] # we will store all the rows with NaN for an attribute in this list

for i in range(len(headers)):
    if (i==0 or i==1):
        continue
    else:
        a = new_df[new_df[headers[i]].isnull()].index
        index.append(a)
    
def rmse_calc(list1,list2,a):
    summ=0
    for i in range(len(index[a])):
        summ=summ+((list1[index[a][i]]-list2[index[a][i]])**2)
    rmse=summ/len(index[a])
    rmse=rmse**0.5
    return rmse


#---------Using replacement by mean------------#

new_df_mean = df.copy()

for i in range(len(headers)):
    if (i==0 or i==1):
        continue
    else:
        new_df_mean[headers[i]].fillna(new_df_mean[headers[i]].mean(),inplace=True)

xvalmean = []
yvalmean = []
        

for i in range(len(headers)):
    if (i==0 or i==1):
        continue
    else:
        print("*****************************************************************************************************************************")
        print(headers[i])
        print("*****************************************************************************************************************************")
        print("The mean for " + headers[i] + " attribute in the database filled with mean is = " + str(new_df_mean[headers[i]].mean()))
        print("The mean for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].mean()))
        print()
        print("The median for " + headers[i] + " attribute in the database filled with mean is = " + str(new_df_mean[headers[i]].median()))
        print("The median for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].median()))
        print()
        print("The mode for " + headers[i] + " attribute in the database filled with mean is = " + str(new_df_mean[headers[i]].mode()[0]))
        print("The mode for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].mode()[0]))
        print()
        rms1=rmse_calc(orig_df[headers[i]],new_df_mean[headers[i]],i-2)
        print("The value of RMSE for the attribute "+ headers[i] + " for the original vs database filled with mean = ",rms1)
        xvalmean.append(headers[i])
        yvalmean.append(rms1)
        print("*****************************************************************************************************************************")
        print()

plt.bar(xvalmean,yvalmean, edgecolor = 'red')
plt.xticks(rotation = 90,size = 10)
plt.show()
#-------Using linear interpolation-------------#

new_df_inter = df.copy()

for i in range(len(headers)):
    if (i==0 or i==1):
        continue
    else:
        new_df_inter[headers[i]].interpolate(method = 'linear',limit_direction = 'forward',inplace = True)

xvalinter = []
yvalinter = []
        
for i in range(len(headers)):
    if (i==0 or i==1):
        continue
    else:
        print("*****************************************************************************************************************************")
        print(headers[i])
        print("*****************************************************************************************************************************")
        print("The mean for " + headers[i] + " attribute in the database filled using interpolation is = " + str(new_df_inter[headers[i]].mean()))
        print("The mean for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].mean()))
        print()
        print("The median for " + headers[i] + " attribute in the database filled using interpolation is = " + str(new_df_inter[headers[i]].median()))
        print("The median for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].median()))
        print()
        print("The mode for " + headers[i] + " attribute in the database filled using interpolation is = " + str(new_df_inter[headers[i]].mode()[0]))
        print("The mode for " + headers[i] + " attribute in the original database is = " + str(orig_df[headers[i]].mode()[0]))
        print()
        rms2 = rmse_calc(orig_df[headers[i]],new_df_inter[headers[i]],i-2)
        print("The value of RMSE for the attribute "+ headers[i] + " for the original vs database filled using interpolation = ",rms2)
        xvalinter.append(headers[i])
        yvalinter.append(rms2)
        print("*****************************************************************************************************************************")
        print()
        
plt.bar(xvalinter,yvalinter,color = "red" , edgecolor = "black")
plt.xticks(rotation = 90,size = 10)
plt.show()


#Question 5
#Part A

rain = new_df_inter['rain'].tolist()
temp = new_df_inter['temperature'].tolist()

def qu5(list,istr):
    q1 = np.percentile(list,25)
    q2 = np.percentile(list,50)
    q3 = np.percentile(list,75)
    iqr = q3-q1
    print("The iqr for the attibute " + istr + " = " + str(iqr))
    print("The median for the attibute " + istr + " = " + str(q2))
    high = q3 + (1.5*iqr)
    low = q1 - (1.5*iqr)
    out_in = []
    out_val =[]
    for i in range (len(list)):
        if (list[i]>high or list[i]<low):
            out_in.append(i)
            out_val.append(list[i])
    plt.boxplot(list,notch = "True" , patch_artist= "True")
    plt.title ("Boxplot of "+istr)
    plt.xlabel(istr)
    plt.show()
    print("The number of outliers = "+ str(len(out_in)))
    print("The outliers are: ")
    print(out_val)

    
print("*****************************************************************************************************************************")
print("temperature")
qu5(temp,"temperature")
print()
print("*****************************************************************************************************************************")

print("*****************************************************************************************************************************")
print("Rain")
qu5(rain,"rain")
print()
print("*****************************************************************************************************************************")


median_r=new_df_inter["rain"].median()
median_t=new_df_inter["temperature"].median()

print("-----------------Question 5(b)-----------------")

new_df_inter["rain"] = np.where(new_df_inter["rain"] >75, median_r,new_df_inter['rain'])
new_df_inter["temperature"] = np.where(new_df_inter["temperature"] >75, median_t,new_df_inter['temperature'])

plt.boxplot(new_df_inter["temperature"],notch = "True" , patch_artist= "True")
plt.title("Box plot of temperature after changing outliers with median")
plt.xlabel("temperature")
plt.show()
plt.boxplot(new_df_inter["rain"],notch = "True" , patch_artist= "True")
plt.title("Box plot of rain after changing outliers with median")
plt.xlabel("rain")
plt.show()
