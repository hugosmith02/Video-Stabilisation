# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:27:38 2023

@author: Hugo
"""

import numpy as np
import cv2
from statsmodels.nonparametric.smoothers_lowess import lowess

def movingAverage(curve, radius):
    '''
    A moving average filter to smooth trajectories. Intensity
    of smoothing defined by radius.
    '''
    #Set number of points to average over
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def gaussianAverage(curve,radius):
    '''
    A gaussian filter to smooth trajectories. Intensity
    of smoothing defined by radius.
    '''
    #Set number of points to apply gaussian filter 
    window_size = 2 * radius + 1
    #Set the standard deviation of tha gaussian curve    
    sigma=radius
    #Define array of window size
    g=np.arange(0,window_size)
    #Define Gaussian Filter
    f=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(g-(window_size/2))**2/(2*sigma**2))
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    #Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory,average,radius):
    #Create copy of trajectory.
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = average(trajectory[:,i], radius)
    return smoothed_trajectory

def lowess_smooth(trajectory,frames,radius):
    #Create copy of trajectory.
    smoothed_trajectory = np.copy(trajectory)
    #Define number of frames
    f=np.size(frames)
    # Filter the x, y and angle curves
    for i in range(0,3):
        #Apply LOWESS filter
        smoothed_trajectory[:,i] = lowess(trajectory[:,i],frames, frac=radius/f)[:,1]
    return smoothed_trajectory

def fixBorder(frame):
    #Define shape of frame
    s = frame.shape
    # Scale the image 8% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.08)
    #Define scaled frame
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Import input video
cap = cv2.VideoCapture('ciggy.mp4')
 
# Extract frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
# Extract width and height of input video
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Read first frame
_, prev = cap.read() 
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)

#Filter over frames
for i in range(n_frames-2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
   
    # Read next frame
    success, curr = cap.read()
    #Break loop at end of video
    if not success:
      break
   
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
   
    # Calculate optical flow by tracking feature points
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
   
    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
   
    #Find the transformation matrix
    m = cv2.estimateAffine2D(prev_pts, curr_pts) #will only work with OpenCV-3 or less
   
    # Extract traslation
    tr=m[0]
    
    #Assign x and y
    dx = tr[0,2]
    dy = tr[1,2]
   
    # Extract rotation angle
    da = np.arctan2(tr[1,0], tr[0,0])
   
    # Store transformation in array
    transforms[i] = [dx,dy,da]
   
    # Move to next frame
    prev_gray = curr_gray
   
    print("Frame: " + str(i) +  "/" + str(n_frames-3) + " -  Tracked points : " + str(len(prev_pts)))
    
#%%
# Take a cumulative summation of the transforms to calculate trajectory
trajectory = np.cumsum(transforms, axis=0)

#Define smoothing radius (ie intensity of smoothing)
smoothing_radius=50

#Form array of 0 to number of frames
frames=np.arange(0,np.size(trajectory[:,0]))

#Smooth trajectory with one of the free filters
smoothed_trajectory=smooth(trajectory,gaussianAverage,smoothing_radius)
#smoothed_trajectory=smooth(trajectory,movingAverage,smoothing_radius)
#smoothed_trajectory=lowess_smooth(trajectory,frames,smoothing_radius)


# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory

# Calculate new transformation array
transforms_smooth = transforms + difference

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
    # Read next frame
    success, frame = cap.read()
    #If end of video break loop
    if not success:
        break
 
    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]
    
    
    # Form new transformation matrix with smooth transformations
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
 
    # Apply affine transformation to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w,h))
    
    # Fix black border artifacts for stabilised video
    frame_stabilized = fixBorder(frame_stabilized)
    #Resize
    frame_stabilized = cv2.resize(frame_stabilized, (540,960))
    
    #Apply fixBorder function to unstabilised video so that the output videos have the same zoom
    frame =fixBorder(frame)
    #resize
    frame = cv2.resize(frame, (540,960))
    #Merge videos together
    frame_out = cv2.hconcat([frame, frame_stabilized])
    #Display video
    cv2.imshow('Unstable and Stable',frame_out)
    cv2.waitKey(10)