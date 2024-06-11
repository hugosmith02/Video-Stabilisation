# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:18:57 2023

@author: Hugo
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#Initialise count of frame numbers
count = 0

#Import video
cap = cv2.VideoCapture("test.mov")
#Extract frame rate
frameRate = cap.get(5)

#Loop over frames in video
while(cap.isOpened()):
    print(count)
    #Extract current frame number
    frameId = cap.get(1)
    #Read the frame
    ret, frame = cap.read()
    #If video is over break loop
    if (ret != True):
        break
    #Define filename to save frame as jpg
    filename ="frame%d.jpg" % count;count+=1
    #Save frame as jpg
    cv2.imwrite(filename, frame)

cap.release()
print ("Done!")

#%%
#Import first frame
img0=plt.imread('frame0.jpg')

#Extract number of rows, columns and colours
rows, cols, ch = img0.shape

#Split image into RGB components
r0=img0[:,:,0]
g0=img0[:,:,1]
b0=img0[:,:,2]

#Define RGB constraints for object/colour of interest.
rup=100
rlo=10
gup=100
glo=10
bup=200
blo=50

#Pick out the pixels that satisfy the constraints, these are the marker pixels
spot0=np.argwhere((r0>=rlo) & (g0>=glo) & (b0>=blo) & (r0<=rup) & (g0<=gup) & (b0<=bup))

#Assign intial y and x values of the marker
y0=spot0[:,0]
x0=spot0[:,1]
    
#Find the centre of mass pixel values for the markers initial position
xavg0=np.round((np.mean(x0)),0)
yavg0=np.round((np.mean(y0)),0)

#Initiate figure
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

#Loop through frames
for i in range(0,count):
    #Clear axis
    ax.cla()
    
    #Import frame
    img=plt.imread('frame%d.jpg' % i)
    
    #Split image into RGB components
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    
    #Pick out the pixels that satisfy the constraints, these are the marker pixels
    spot=np.argwhere((r>=rlo) & (g>=glo) & (b>=blo) & (r<=rup) & (g<=gup) & (b<=bup))
    
    #Assign y and x values of the marker
    y=spot[:,0]
    x=spot[:,1]
    
    #Find the centre of mass pixel values for the marker
    xavg=np.round((np.mean(x)),0)
    yavg=np.round((np.mean(y)),0)
    
    #Calculate the difference in x and y between the new
    #pixel position and the initial position.
    ydiff=yavg-yavg0
    xdiff=xavg-xavg0
    
    #Define transformation matrix
    M=np.array([[1,0,-xdiff],[0,1,-ydiff]])
    
    #Applying affine transformation
    frame_new = cv2.warpAffine(img, M,(cols,rows))
    
    #Plot the translated frame
    ax.imshow(frame_new)
    
    #Pause between each frame
    plt.pause(.01)