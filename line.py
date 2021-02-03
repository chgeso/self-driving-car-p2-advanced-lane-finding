import numpy as np
import cv2

class line():

    def __init__(self, nWindows = 9, Margin = 100, Minpix = 50):
        # Choose the number of sliding windows
        self.nwindows = nWindows

        # Set the width of the windows +/- margin
        self.margin = Margin

        # Set minimum number of pixels found to recenter window
        self.minpix = Minpix

    # Find Lane Pixels
    def find_lane_pixels(self, warped):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((warped, warped, warped))

        # Find the peak of the left and right halves of the histogram.
        # These will be the starting point for the left and right lines.
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = self.nwindows
        margin = self.margin
        minpix = self.minpix

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(warped.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height

            # Find the four boundaries of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found over minpix pixels, recenter next window ('right' or 'leftx_current') on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previosuly was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            print('ValueError caused')
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, warped, leftx, lefty, rightx, righty, out_img):

        # Fit a second order polynomial to each using 'np.polyfit'
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        yvals = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        try:
            left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
            right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
        except TypeError:
            # Avoids an error if 'left' and 'right_fit' are still none or incorrect.
            print('The line function failed to fit a line!')
            left_fitx = 1*yvals**2 + 1*yvals
            right_fitx = 1*yvals**2 + 1*yvals

        # Corlors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        left_fitx = np.array(left_fitx,np.int32)
        right_fitx = np.array(right_fitx,np.int32)

        return left_fitx, right_fitx, yvals, out_img

    def measure_curvature_pixels(self, leftx, lefty, rightx, righty, yvals, ym_per_pix, xm_per_pix):

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        y_eval_left = np.max(lefty)*ym_per_pix
        y_eval_right = np.max(righty)*ym_per_pix

        # Implement the calculation of R curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad



