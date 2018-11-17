#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:34:18 2017
Modified on Jan 6 2018 for Leap data

@author: Wenjin
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage



def normalize_data(original_data):
    """
    Return the normalized images
    With color values in 0.1~0.9
    """
    a = 0.
    b = 1.
    
    Xmin = np.min(original_data)
    Xmax = np.max(original_data)
    
    norm_data = np.empty_like(original_data, dtype=np.float32)
    norm_data = (original_data-Xmin)*(b-a)/(Xmax-Xmin) + a
    return norm_data

def extract_depth(img, d_range=120):
    """
    Return the depth image
    With depth range in [1, d_range+1] and 0 for other pixels 
    """
    # Find the min distance to the camera
    d_indices = np.where(img>0)
    d_min = np.min(img[d_indices])
    # Set the range for hand segmentation    
    img[d_indices] = img[d_indices] - d_min + 1    
    img[img > d_range] = 0
    
    # Reverse
    reverse_mask = (img>0)*(d_range+1)
    img = reverse_mask - img
   
    return img

def crop_hand(img, thresh=50):
    dist_x = np.sum(img, 0)
    dist_y = np.sum(img, 1)
    span_x = np.where(dist_x>thresh)
    span_x_start = np.min(span_x)
    span_x_end = np.max(span_x)

    span_y = np.where(dist_y>thresh)
    span_y_start = np.min(span_y)
    span_y_end = np.max(span_y)
    
    return img[span_y_start:span_y_end+1, span_x_start:span_x_end+1]

def polar_hist(theta_hist):
    N = 72
    bottom = 200
#    max_height = 4

    theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
#     radii = max_height*np.random.rand(N)
    width = (2*np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, theta_hist, width=width, bottom=bottom)
    
    ax.set_xticklabels(['0', '', 'pi/2', '', 'pi', '', '-pi/2', ''])
    ax.set_yticklabels('')

    # Use custom colors and opacity
    for r, bar in zip(theta_hist, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.8)

    plt.show()
    
def cal_cluster(bins, theta_hist, thresh=50):
    center = (bins[:-1] + bins[1:]) / 2
    n_bins = len(bins)-1
    step = np.pi*2/n_bins
    
    # find those positions where theta_hist larger than a threshold
    X = center[theta_hist>thresh]
    threshed_theta_hist = theta_hist[theta_hist>thresh]   
    
    if len(X) > 1:    
        # split into clusters
        neighbor_diff = X[1:]-X[:-1]    
        split_indices = np.where(neighbor_diff>np.pi*2/n_bins+0.01)[0] # should add 1 to index X
        
        if len(split_indices)>0:
            split_indices = [ele+1 for ele in split_indices]
            n_clusters = len(split_indices)+1
            
            # take care of the boundary between -pi and pi
            if (X[0]+np.pi-step/2)**2 + (X[-1]-np.pi+step/2)**2 < 1e-3:
        #         print('boundary fusion')
                
                # move the first cluster to the end
                X_to_end = X[:split_indices[0]]
                X_to_end = [ele+2*np.pi for ele in X_to_end]
                X = X[split_indices[0]:].tolist() + X_to_end
                
                threshed_theta_hist_to_end = threshed_theta_hist[:split_indices[0]]
                threshed_theta_hist = threshed_theta_hist[split_indices[0]:].tolist() + threshed_theta_hist_to_end.tolist()
                
                n_clusters -= 1
                idx_adjust = split_indices[0]
                split_indices = [ele-idx_adjust for ele in split_indices[1:]]
            
            # calculate mean theta and sum of pixels of each culster
            cluster_mean_list = []
            cluster_sum_list = []
            cluster_range_list = []
            cluster_start_idx = 0
            for split_idx in split_indices:
                cluster_end_idx = split_idx
                cluster_mean_list.append(np.mean(X[cluster_start_idx:cluster_end_idx]))
                cluster_sum_list.append(np.sum(threshed_theta_hist[cluster_start_idx:cluster_end_idx]))
                cluster_range_list.append([X[cluster_start_idx], X[cluster_end_idx-1]])
                cluster_start_idx = cluster_end_idx
            cluster_mean_list.append(np.mean(X[cluster_start_idx:]))
            cluster_sum_list.append(np.sum(threshed_theta_hist[cluster_start_idx:]))
            cluster_range_list.append([X[cluster_start_idx], X[-1]])
            
            # to numpy array
            cluster_mean_list = np.array(cluster_mean_list)
            cluster_sum_list = np.array(cluster_sum_list)
            cluster_range_list = np.array(cluster_range_list)
            
            # sorting
            sort_indices = np.argsort(-cluster_sum_list)
            cluster_mean_list = cluster_mean_list[sort_indices]
            cluster_sum_list = cluster_sum_list[sort_indices]
            cluster_range_list = cluster_range_list[sort_indices]
            
            # range adjustment
            cluster_mean_list = [ele-2*np.pi if ele>np.pi else ele for ele in cluster_mean_list]
            cluster_sum_px_list = np.copy(cluster_sum_list)
            # normalization
            cluster_sum_list = cluster_sum_list/np.sum(cluster_sum_list)
            
        else: # only one cluster
            n_clusters = 1
            cluster_mean_list = [np.mean(X)]
            cluster_sum_list = [1.]
            cluster_sum_px_list = [np.sum(threshed_theta_hist)]
            
            # to numpy array
            cluster_mean_list = np.array(cluster_mean_list)
            cluster_sum_list = np.array(cluster_sum_list) 

            # range adjustment
            cluster_mean_list = [ele-2*np.pi if ele>np.pi else ele for ele in cluster_mean_list]
            
            cluster_range_list = np.array([X[0], X[-1]])
        
        
    else: # no cluster
        n_clusters = 0
        split_indices = []
        cluster_mean_list = [np.pi/2]
        cluster_sum_list = [3]
        cluster_sum_px_list = [0]
        cluster_range_list = []
    
    
    return n_clusters, cluster_mean_list, cluster_sum_list, cluster_sum_px_list, cluster_range_list

def remove_noise(noise_img, area_thresh=200):
    # convert to 8bit image
    img = np.zeros_like(noise_img, dtype=np.uint8)
    img[noise_img>0]=1
    
    # refer to: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
#     min_size = 200  

    #your answer image
    mask = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= area_thresh:
            mask[output == i + 1] = 1
            
    return np.multiply(mask, noise_img)#, sizes


def rotate_bound(image, angle):
    # http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def debug_info(info):
    print(info)

def preprocess_depth_img(depth_img, d_range=120, palm_radius_thresh=45, cluster_thresh=20, arm_px_thresh=1000, rotation=False, debug=False):
    """
    Params:
        depth_img, 
        d_range=120, 
        palm_radius_thresh
        cluster_thresh
        rotation=False
        debug=False
    
    """
    if d_range is not None:
        depth_img = extract_depth(depth_img, d_range) 

    # remove noise/small island
    if depth_img.shape[1]>250:
        depth_img = remove_noise(depth_img, area_thresh=200)

    depth_img = crop_hand(depth_img)

    depth_img_result = np.copy(depth_img)
    depth_img_debug = np.stack([np.copy(depth_img),np.copy(depth_img),np.copy(depth_img)],2)

    # Mass center
    mc = ndimage.measurements.center_of_mass(depth_img)
    # draw a larger circle to remove the palm
    img_dim = np.min(depth_img.shape)
    refer_palm_radius = img_dim//2
    palm_radius = np.min([refer_palm_radius, palm_radius_thresh])
    cv2.circle(depth_img, (int(mc[1]), int(mc[0])), palm_radius, [0,0,0], thickness=-1, lineType=8, shift=0) 

    
    # find the nonzero points
    p_Y, p_X = np.where(depth_img>0)
    # calculate the theta value of each point
    theta = np.arctan2(-p_Y+mc[0], p_X-mc[1])
    # hist
    bins = np.linspace(-np.pi, np.pi, 73)
    theta_hist = np.histogram(theta, bins=bins)

    
#     canvas = np.stack([depth_img_canvas, depth_img_canvas, depth_img_canvas], 2)



    # Calculate clusters
    n_clusters, cluster_mean_list, cluster_sum_list, cluster_sum_px_list, cluster_range_list = \
                                                    cal_cluster(bins, theta_hist[0], cluster_thresh)
    
    arm_mask = np.zeros_like(depth_img)
    
    get_a_valid = False # valid cluster as arm
    cluster_range = []
    
    if n_clusters == 0:
        cluster_mean = -np.pi/2 
        cluster_sum = 3 # no cluster
        cluster_sum_px = 0
        
    elif n_clusters == 1:
        if cluster_mean_list[0]<np.pi/6 or cluster_mean_list[0]>np.pi/6*5:
            cluster_mean = cluster_mean_list[0]
            cluster_sum = cluster_sum_list[0]
            cluster_sum_px = cluster_sum_px_list[0]
            cluster_range = cluster_range_list
            get_a_valid = True
        else:
            cluster_mean = -np.pi/2
            cluster_sum = 2 # 1 cluster but in the wrong region
            cluster_sum_px = 0
            
    else: # has more than 1 clusters
#        get_a_valid = False
        for i in range(n_clusters):
            if cluster_mean_list[i]<np.pi/6 or cluster_mean_list[i]>np.pi/6*5:                
                cluster_mean = cluster_mean_list[i]
                cluster_sum = cluster_sum_list[i]
                cluster_sum_px = cluster_sum_px_list[i]
                cluster_range = cluster_range_list[i]
                get_a_valid = True
                break
        if get_a_valid == False:
            cluster_mean = -np.pi/2 
            cluster_sum = 4 # no cluster
            cluster_sum_px = 0
            
#    if debug:
#        debug_info(cluster_range)

    # create the arm mask
    theta_in_range_indices = []
    if get_a_valid:
        cluster_range = cluster_range.ravel()
        # take care of the boundary between pi and -pi
        if cluster_range[1]<np.pi:
            theta_in_range_indices = ((theta>cluster_range[0]-np.pi/18) & (theta<cluster_range[1]+np.pi/18))
        else:
            theta_in_range_indices_part1 = ((theta>cluster_range[0]-np.pi/18) & (theta<np.pi))
            theta_in_range_indices_part2 = ((theta>=-np.pi) & (theta<cluster_range[1]-2*np.pi+np.pi/18) )
            theta_in_range_indices = theta_in_range_indices_part1 | theta_in_range_indices_part2
        
    p_X_in_range = p_X[theta_in_range_indices]
    p_Y_in_range = p_Y[theta_in_range_indices]  
        
    for p_y, p_x in zip(p_Y_in_range, p_X_in_range):
        arm_mask[p_y, p_x]  = 1
    
     
    rot_angle = cluster_mean*180/np.pi+90
#     M_r = cv2.getRotationMatrix2D((int(mc[1]), int(mc[0])), rot_angle, 1)

    if cluster_sum_px > arm_px_thresh:
        depth_img_result = depth_img_result*(1-arm_mask)
    
    if rotation:
        depth_img_result = rotate_bound(depth_img_result, rot_angle)
    # crop once more
    depth_img_result = crop_hand(depth_img_result)
    
    ## DEBUG
    if debug:
        cv2.circle(depth_img_debug, (int(mc[1]), int(mc[0])), palm_radius, [0,0,255], thickness=1, lineType=8, shift=0)
        # Draw hist
        center = (bins[:-1] + bins[1:]) / 2
        for alpha, val in zip(center, theta_hist[0]):
            cv2.line(depth_img_debug, (int(mc[1]), int(mc[0])),
                     (int(mc[1]+0.15*val*np.cos(alpha)), int(mc[0]-0.15*val*np.sin(alpha))),
                     [255,0,0], thickness=1, lineType=8, shift=0)
            
        # Draw arm direction
        cv2.arrowedLine(depth_img_debug, (int(mc[1]), int(mc[0])), 
                        (int(mc[1]+100*cluster_sum*np.cos(cluster_mean)), int(mc[0]-100*cluster_sum*np.sin(cluster_mean))),
                        [0,0,255], thickness=3, line_type=8, shift=0) # y axis is reversed
        cv2.putText(depth_img_debug,str(cluster_sum)[0:4], (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255])
        cv2.putText(depth_img_debug,str(cluster_sum_px), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255])        
        
        # Highlight the detected arm region
        if cluster_sum_px > arm_px_thresh:
            depth_img_debug[:,:,1] = depth_img_debug[:,:,1]*(1-arm_mask) + depth_img_debug[:,:,1]*arm_mask*255
        
        return depth_img_debug
    
    else:
        return depth_img_result
    
    
def resize_img(img, target_size=32):
    
    if len(img.shape)>2:
        height, width, channels = img.shape

        if height >= width:
            img_result = np.zeros((height, height, channels), np.float32)
            col_start = (height-width)//2
            img_result[:, col_start:col_start+width, :] = img
        else:
            img_result = np.zeros((width, width, channels), np.float32)
            row_start = (-height+width)//2
            img_result[row_start:row_start+height, :, :] = img
    else:
        height, width = img.shape

        if height >= width:
            img_result = np.zeros((height, height), np.float32)
            col_start = (height-width)//2
            img_result[:, col_start:col_start+width] = img
        else:
            img_result = np.zeros((width, width), np.float32)
            row_start = (-height+width)//2
            img_result[row_start:row_start+height, :] = img
    
    return cv2.resize(img_result, (target_size, target_size))    
