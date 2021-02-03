import numpy as np
import cv2

class threshold():
    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    def abs_sobel_thresh(self, img, orient='x', thresh=(0,255)):
    
        # Apply the following steps to img
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # Take the derivative in x or y given orient = 'x' or 'y'
        # Take the absolute value of the derivative or gradient

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        else:
            print('You have to choose x or y')

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient and applies a threshold.
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0,255)):

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)

        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)

        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):

        # Grayscaling
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel) 

        # Take the absolute value of the gradient direction, apply a threshold, and create a binary image result.
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return binary_output

    # Define a function that applies S channel of HLS.
    # then applies a threshold.
    def color_threshold(self, img, sthresh=(0,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

        binary_output = s_binary
    
        return binary_output