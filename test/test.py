#%matplotlib tk
#import matplotlib.pyplot as plt
#import numpy as np
#from drawnow import *

#fig = plt.figure(1)
 
#angles = []
#sines = []
#cosines = []
 
#def show_plot():
#    plt.plot(angles,sines,label='Sine')
#    plt.plot(angles,cosines,label='Cosine')
#    plt.legend()
#    plt.grid()
#    plt.xlabel('Angles [deg]')
#    plt.ylabel('Value')

#for x in np.linspace(0,np.pi*2,100):
#    angles = np.append(angles, x)
#    sines  = np.append(sines, np.sin(x))
#    cosines= np.append(cosines, np.cos(x))
#    drawnow(show_plot)

#import random
#from itertools import count
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
 
#plt.style.use('fivethirtyeight')
 
#x_val = []
#y_val = []
 
#index = count()
 
#def animate(i):
#    x_val.append(next(index))
#    y_val.append(random.randint(0,5))
#    plt.cla()
#    plt.plot(x_val, y_val)
 
#ani = FuncAnimation(plt.gcf(), animate, interval = 1)
 
#plt.tight_layout()
#plt.show()

#import numpy as np
#import matplotlib.pyplot as plt

#x = 0
#for i in range(1000):
#    x = x + 0.1
#    y = np.sin(x)

#    plt.scatter(x, y)
#    plt.pause(0.001)

#plt.show()

#==============================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as ticker
import cv2
import glob

arr_x = np.array([0, 1, 2, 3, 4, 5])
arr_y = np.array([6,12,16,12,16,12])
arr_c = np.array([6,12,16,12,16,12])

plt.xlabel("time(s)")
plt.xlim([0, 7]) # X축의 범위
plt.ylim([0, 20]) # Y축의 범위

model = make_interp_spline(arr_x, arr_y)

count = 0

#ax = plt.axes()
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

xs = np.array([])
ys = np.array([])
plt.plot(xs, ys, 'b')

point_x = np.array([])
point_y = np.array([])
sw = 0

a = 100
b = 1
for i in np.linspace(0,6,a*b):
    mx = i
    my = model(mx)
    xs = np.append(xs, np.array([mx]))
    ys = np.append(ys, np.array([my]))
    plt.cla()
    if i >= (6 / a * b) * (b*8) and sw == 0:
        sw = sw + 1
        print("1")
        point_x = np.append(point_x, np.array(xs[-1]))
        point_y = np.append(point_y, np.array(ys[-1]))
    elif i >= (6 / a * b) * (b*18) and sw == 1:
        sw = sw + 1
        print("2")
        point_x = np.append(point_x, np.array(xs[-1]))
        point_y = np.append(point_y, np.array(ys[-1]))
    elif i >= (6 / a * b) * (b*31) and sw == 2:
        sw = sw + 1
        print("3")
        point_x = np.append(point_x, np.array(xs[-1]))
        point_y = np.append(point_y, np.array(ys[-1]))
    elif i >= (6 / a * b) * (b*52) and sw == 3:
        sw = sw + 1
        print("4")
        point_x = np.append(point_x, np.array(xs[-1]))
        point_y = np.append(point_y, np.array(ys[-1]))
    elif i >= (6 / a * b) * (b*71) and sw == 4:
        sw = sw + 1
        print("5")
        point_x = np.append(point_x, np.array(xs[-1]))
        point_y = np.append(point_y, np.array(ys[-1]))
    else :
        pass

    if ys.size == 0 :
        pass
    else :
         y = ys[-1]
    if y > 0:
        print(i)
        plt.plot(xs, ys, 'b')
        plt.savefig('C:/Users/user/Desktop/data/savefig_'+str(int(count))+'.png')
        count = count + 1
    else :
        pass

print(count)
plt.show()

img_array = []
count1 = 0
for count1 in range(count):
    filename = "C:/Users/user/Desktop/data/savefig_"+ str(count1) +".png"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('C:/Users/user/Desktop/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
#==============================================================

#import matplotlib.pyplot as plt
#import numpy as np
#import time
#from drawnow import *
  
##plt.xlim([-1.5, 1.5]) # X축의 범위
#plt.ylim([-1.5, 1.5]) # Y축의 범위

#count = 0
#angles = np.array([])
#sines = np.array([])
#for x in np.linspace(0,np.pi*29,200):
#    weight = 0.75
#    angles = np.append(angles, x)
#    sines  = np.append(sines, np.sin(x * weight))
#    plt.cla()
#    plt.plot(angles, sines, 'b')
#    plt.savefig('C:/Users/user/Desktop/cos_data1/savefig_'+str(int(count))+'.png')
#    count = count + 1

#print("1")
#count = 0
#angles = np.array([])
#sines = np.array([])
#for x in np.linspace(0,np.pi*29,200):
#    weight = 1
#    angles = np.append(angles, x)
#    sines  = np.append(sines, np.sin(x * weight))
#    plt.cla()
#    plt.plot(angles, sines, 'b')
#    plt.savefig('C:/Users/user/Desktop/cos_data2/savefig_'+str(int(count))+'.png')
#    count = count + 1

#print("2")
#count = 0
#angles = np.array([])
#sines = np.array([])
#for x in np.linspace(0,np.pi*29,200):
#    weight = 0.75
#    angles = np.append(angles, x)
#    sines  = np.append(sines, np.sin(x * weight))
#    plt.cla()
#    plt.plot(angles, sines, 'b')
#    plt.savefig('C:/Users/user/Desktop/cos_data3/savefig_'+str(int(count))+'.png')
#    count = count + 1

#print("3")
#count = 0
#angles = np.array([])
#sines = np.array([])
#for x in np.linspace(0,np.pi*29,200):
#    weight = 1.25
#    angles = np.append(angles, x)
#    sines  = np.append(sines, np.sin(x * weight))
#    plt.cla()
#    plt.plot(angles, sines, 'b')
#    plt.savefig('C:/Users/user/Desktop/cos_data4/savefig_'+str(int(count))+'.png')
#    count = count + 1
#plt.show()

#import matplotlib.pyplot as plt
#import numpy as np
#from scipy.interpolate import make_interp_spline
#import matplotlib.ticker as ticker
#import cv2
#import glob

#print("")
#img_array = []
#count1 = 0
#for count1 in range(200):
#    filename = "C:/Users/user/Desktop/cos_data1/savefig_"+ str(count1) +".png"
#    img = cv2.imread(filename)
#    height, width, layers = img.shape
#    size = (width,height)
#    img_array.append(img) 
 
#out = cv2.VideoWriter('C:/Users/user/Desktop/project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()

#img_array = []
#count1 = 0
#for count1 in range(200):
#    filename = "C:/Users/user/Desktop/cos_data2/savefig_"+ str(count1) +".png"
#    img = cv2.imread(filename)
#    height, width, layers = img.shape
#    size = (width,height)
#    img_array.append(img) 
 
#out = cv2.VideoWriter('C:/Users/user/Desktop/project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()

#img_array = []
#count1 = 0
#for count1 in range(200):
#    filename = "C:/Users/user/Desktop/cos_data3/savefig_"+ str(count1) +".png"
#    img = cv2.imread(filename)
#    height, width, layers = img.shape
#    size = (width,height)
#    img_array.append(img) 
 
#out = cv2.VideoWriter('C:/Users/user/Desktop/project3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()

#img_array = []
#count1 = 0
#for count1 in range(200):
#    filename = "C:/Users/user/Desktop/cos_data4/savefig_"+ str(count1) +".png"
#    img = cv2.imread(filename)
#    height, width, layers = img.shape
#    size = (width,height)
#    img_array.append(img) 
 
#out = cv2.VideoWriter('C:/Users/user/Desktop/project4.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()