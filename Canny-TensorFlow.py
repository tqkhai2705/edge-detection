# Implementation of Canny Edge Detection in TensorFlow
#	--> Reference: https://en.wikipedia.org/wiki/Canny_edge_detector
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

GAUS_KERNEL = 3
GAUS_SIGMA  = 1.0
def Gaussian_Filter(kernel_size=GAUS_KERNEL, sigma=GAUS_SIGMA): #Default: Filter_shape = [5,5]
# 	--> Reference: https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter
	k = (kernel_size-1)/2 
	filter = []
	for i in range(kernel_size):
		filter_row = []
		for j in range(kernel_size):
			Hij = np.exp(-((i+1-(k+1))**2 + (j+1-(k+1))**2)/(2*sigma**2))/(2*np.pi*sigma**2)
			filter_row.append(Hij)
		filter.append(filter_row)
	
	return np.asarray(filter).reshape(kernel_size,kernel_size,1,1)

"""
 NOTE: 	All variables are initialized first for reducing proccessing time.
		(If needed) Please remove them and uncomment the corresponding lines in TF_Canny function.
"""
gaussian_filter = tf.constant(Gaussian_Filter(), tf.float32) 						#STEP-1
h_filter = tf.reshape(tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], tf.float32), [3,3,1,1])	#STEP-2
v_filter = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], tf.float32), [3,3,1,1])	#STEP-2

np_filter_PHI = np.zeros((3,3,1,4))
np_filter_PHI[0,0,0,0] = 1 #first pixel is top/left, passed to the first channel
np_filter_PHI[2,2,0,1] = 1 #second pixel is bottom/right, passed to the second channel
np_filter_PHI[0,2,0,2] = 1 #third pixel is top/right, passed to the third channel
np_filter_PHI[2,0,0,3] = 1 #fourth pixel is bottom/left, passed to the fourth channel
filter_PHI = tf.constant(np_filter_PHI, tf.float32)										#STEP-3
np_filter_G = np.zeros((3,3,1,4))
np_filter_G[0,1,0,0], np_filter_G[2,1,0,1], np_filter_G[1,0,0,2], np_filter_G[1,2,0,3] = 1,1,1,1 # Top-Bottom-Left-Right
filter_G = tf.constant(np_filter_G, tf.float32)											#STEP-3

def TF_Clip(x, min, max): return tf.clip_by_value(x, min, max)

""" The Input must have the shape [1, width, height, 1] """
def TF_Canny(img_tensor, minRate=0.05, maxRate=0.2, remove_high_val=False):
	MAX = tf.reduce_max(img_tensor)
	""" STEP-1: Noise reduction with Gaussian filter """
	# gaussian_filter = tf.constant(Gaussian_Filter(kernel_size=3, sigma=0.5), dtype=tf.float32)
	# img_tensor /= 255; img_tensor = tf.clip_by_value(img_tensor, 0, 0.1)
	x_gaussian = TF_Clip(tf.nn.convolution(img_tensor, gaussian_filter, padding='SAME'), 0, MAX)
	# Below is a heuristic to remove the intensity gradient inside a cloud
	if remove_high_val: x_gaussian = TF_Clip(x_gaussian, 0, MAX/5)# tf.reduce_mean(x_gaussian))	
	
	
	""" STEP-2: Calculation of Horizontal and Vertical derivatives  with Sobel operator 
		--> Reference: https://en.wikipedia.org/wiki/Sobel_operator	
	"""
	# h_filter = tf.reshape(tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], tf.float32), [3,3,1,1])
	# v_filter = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], tf.float32), [3,3,1,1])
	Gx = tf.nn.convolution(x_gaussian, h_filter, padding='SAME')
	Gy = tf.nn.convolution(x_gaussian, v_filter, padding='SAME')
	G 		= tf.sqrt(tf.square(Gx) + tf.square(Gy))
	BIG_PHI	= tf.atan2(Gy,Gx)
	
	""" STEP-3: NON-Maximum Suppression
		--> Reference: https://stackoverflow.com/questions/46553662/conditional-value-on-tensor-relative-to-element-neighbors
	"""
	""" 3.1-Selecting Edge-Pixels on diagonal directions """
	# np_filter_PHI = np.zeros((3,3,1,4))
	# np_filter_PHI[0,0,0,0] = 1 #first pixel is top/left, passed to the first channel
	# np_filter_PHI[2,2,0,1] = 1 #second pixel is bottom/right, passed to the second channel
	# np_filter_PHI[0,2,0,2] = 1 #third pixel is top/right, passed to the third channel
	# np_filter_PHI[2,0,0,3] = 1 #fourth pixel is bottom/left, passed to the fourth channel
	# filter_PHI = tf.constant(np_filter_PHI, tf.float32)	
	targetPixels_PHI = tf.nn.convolution(BIG_PHI, filter_PHI, padding='SAME')
	isGreater_PHI = tf.cast(tf.greater(BIG_PHI, targetPixels_PHI), tf.float32)
	
	# Merging the 4 channels, considering they're 0 for false and 1 for true
	# Note: Indices [:,:,:,0:1] is for keeping 4 dimensions (Indices [:,:,:,0] will return only 3 dimensions)
	isMax_PHI1 = isGreater_PHI[:,:,:,0:1]*isGreater_PHI[:,:,:,1:2] #Diagonal direction (north-west to south-east)
	isMax_PHI2 = isGreater_PHI[:,:,:,2:3]*isGreater_PHI[:,:,:,3:4] #Diagonal direction (north-east to south-west)
	# Now, the center pixel will remain if isGreater = 1 at that position:
	edges_PHI = TF_Clip(BIG_PHI*isMax_PHI1 + BIG_PHI*isMax_PHI2, 0, MAX)
	
	""" 3.2-Selecting Edge-Pixels on Horizontal and Vertical directions """
	# np_filter_G = np.zeros((3,3,1,4))
	# np_filter_G[0,1,0,0], np_filter_G[2,1,0,1], np_filter_G[1,0,0,2], np_filter_G[1,2,0,3] = 1,1,1,1 # Top-Bottom-Left-Right
	# filter_G = tf.constant(np_filter_G, tf.float32)
	targetPixels_G = tf.nn.convolution(G, filter_G, padding='SAME')
	isGreater_G = tf.cast(tf.greater(G, targetPixels_G), tf.float32)
	isMax_G1 = isGreater_G[:,:,:,0:1]*isGreater_G[:,:,:,1:2] #Vertical direction (top to bottom)
	isMax_G2 = isGreater_G[:,:,:,2:3]*isGreater_G[:,:,:,3:4] #Horizontal direction (left to right)
	edges_G = TF_Clip(G*isMax_G1 + G*isMax_G2, 0, MAX)
	
	""" 3.3-Merging Edges on Horizontal-Vertical and Diagonal directions """
	edges_merged = TF_Clip(edges_G + edges_PHI, 0, MAX)
	
	"""STEP-4: Hysteresis Thresholding
		(I remove values smaller than the average. You may change this threshold.)
		--> The result is a matrix with TRUE (edge pixel) and FALSE (non-edge pixel)
	"""
	# edges_merged /= tf.reduce_max(edges_merged)
	edges_sure = tf.cast(tf.greater(edges_merged, maxRate*MAX), tf.float32)
	edges_weak_and_sure = tf.cast(tf.greater(edges_merged, minRate*MAX), tf.float32)
	
	np_filter_sure = np.ones([3,3,1,1]); np_filter_sure[1,1,0,0] = 0
	filter_sure = tf.constant(np_filter_sure, tf.float32)
	edges_connected_to_sure = tf.nn.convolution(edges_sure, filter_sure, padding='SAME')
	# edges_final = tf.greater(edges_sure + edges_connected_to_sure*edges_weak_and_sure, 0)
	edges_final = edges_sure
	return tf.cast(tf.squeeze(edges_final), tf.float32)

def Edges_Plot(img, edges):
		plt.subplot(121), plt.imshow(img, cmap='nipy_spectral')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122), plt.imshow(edges, cmap='gray')
		plt.title('Edges'), plt.xticks([]), plt.yticks([])
		plt.tight_layout()
		plt.show()
		plt.close('all')
	
if __name__ == '__main__': # Test the above code
	import cv2		# OpenCV is used for comparing final result
	from matplotlib import pyplot as plt
	
	input_img = tf.placeholder(tf.float32, [101,101])
	x_tensor = tf.expand_dims(tf.expand_dims(input_img, axis=0),-1)
	final_output = TF_Canny(x_tensor, remove_high_val=True)
	
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session() as sess:
		sess.run(init_op)		
		img = cv2.imread('examples/example-3.png')[:,:,0]
		edges = cv2.Canny(img, 101, 101)		
		f_img = sess.run(final_output, feed_dict={input_img:img})
		print(f_img.shape)
		print(np.max(f_img))
		print(np.min(f_img))
		Edges_Plot(img, f_img)