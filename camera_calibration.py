import pickle
import cv2
import numpy as np
import glob

# The numbers of x points and y points.
nx = 9
ny = 6

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... , (nx-1,ny-1,0)
# Assuming the chessboard is fixed on the (x, y) plane at z = 0.
# objp is just a replicated array of coordinates.
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Define a function to find the chessboard corners and draw the corners.
def chess_draw(idx, img, nx, ny):
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners.
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If coners found,
    if ret == True:
        # Append the 3d points and 2d points whenever I detect the chessboard corners successfully.
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw corners.
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        # Store the drawing results.
        result_name = 'result'+str(idx)+'.jpg'
        cv2.imwrite('./camera_cal/'+result_name, img)

# Collect objpoints and imgpoints.
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    chess_draw(idx, img, nx, ny)
    if idx == 0:
        # Check the size of image.
        img_size = (img.shape[1], img.shape[0])

# STEP1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save an undistorted image
img = cv2.imread('./camera_cal/calibration1.jpg')
undistorted_img = cv2.undistort(img,mtx,dist,None,mtx)
result_name = 'undistorted_calibration1.jpg'
cv2.imwrite('./camera_cal/'+result_name, undistorted_img)

# Save the calibration result
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open("./calibration_pickle.p", "wb"))