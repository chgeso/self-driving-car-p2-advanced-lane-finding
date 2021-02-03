**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted calibration image"
[image2]: ./camera_cal/undistorted_calibration1.jpg "Undistorted calibration image"
[image3]: ./test_images/test1.jpg "Distorted test image"
[image4]: ./output_images/undistorted_test2.jpg "Undistorted test image"
[image5]: ./output_images/thresholded_test2.jpg "Binary Example"
[image6]: ./output_images/warped_test2.jpg "Warp Example"
[image7]: ./output_images/detected_test2.jpg "Fit Visual"
[image8]: ./output_images/tracked2.jpg "Output"
[video1]: ./tracked_project_video.mp4 "Video"

### Camera Calibration

#### STEP1) Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

For this step, I implemented python codes in `camera_calibration.py`.   
Basically, I defined the numbers of x points and y points to prepare object points.   
By assuming the chessboard is fixed on the (x, y) planed at z = 0, a replicated array of coordinates is defined as `objp`. `objpoints` and `imgpoints` are 3D points in real world space and 2D points in image plane. They are used as arguments to compute the camera clibration matrix and distortion coefficients. To collect objpoints and imgpoints, I defined `chess_draw` function to append the points from the calibration images. After that, I compute `mtx` and `dist` by calling `cv2.calibrateCamera` function. Finally, I save `mtx` and `dist` to `calibration_pickle.p` to reuse in `pipeline`.   

Here are the compared images to show how different to apply `cv2.undistort` function.

![alt text][image1]   
`Distorted calibration image`   
   
![alt text][image2]   
`Undistorted calibration image`   


### Pipeline (single images)

#### STEP2) Provide an example of a distortion-corrected image.

First of all, the pipeline implementation is in `single_image_pipeline.py`.   
To see whether the distortion correction was calculated correctly, I applied `undistort` function with `mtx` and `dist` to the test images.   
Below are an original image and the result of an undistorted test image.   
An original image   
![alt text][image3] 

The undistorted test image     
![alt text][image4]   

#### STEP3) Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I defined `abs_sobel_thresh`, `mag_thresh`, `dir_threshold`, and `color_threshold` in `threshold.py`.
To apply thresholds, I choose `abs_sobel_thresh` with x-orient, `mag_thresh`, and `color_threshold` which is determined by only s channel in HLS.   
Below is the result of processed binary.

![alt text][image5]

#### STEP4) Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To apply a perspective transform to rectify the binary image, I defined `src` and `dst` hardcoded like below :

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
I got `M` and `Minv` parameters by using `getPerspectiveTransform` cv2 function. In this point, `M` is used to do the perspective transform in `warpPerspective` cv2 function.   
Below is the result of warped image.  
   
![alt text][image6]
   
#### STEP5) Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this step, I implemented `line` class in `line.py`. `line` is to detect lane pixels and fit to find the lane boundary. `line` has three functions which are `find_lane_pixels`, `fit_polynomial` and `measure_curvature_pixels`. Firstly, I used `find_lane_pixels` to find pixels by detecting peak of the left and right halves of the historgram and moving the next left or right start points based on a mean value of good left or right indexes. After finding pixels, I fitted a second order polynomial by using `np.polyfit`. From this function, I got left and right fit x values, and y values.   
Below is the result of the detected image.   

![alt text][image7]

#### STEP6) Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For this calculation, I defined `ym_per_pix` and `xm_per_pix` which represent meter per pixel in y or x dimension. The parameters are applied to calculate the radius of curvature of the lane. From `measure_curvature_pixels` function, I got `left_curverad` and `right_curverad`. In this project, I show only `left_curverad`. For calculating the position of the vehicle with respect to center, I subtract `camera_center` and `image cetner`. Of course, I apply `xm_per_pix`.

#### STEP7) Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To draw the left and right lane, I used `fillPoly` cv2 function which draws a polygon based an array of (x,y). Therefore, I collected the left and right (x,y) arrays. I also collected the (x,y) array between the left and right lanes. It was filled with another color. After applying `fillPoly`, in this point, I utilized `Minv` to apply `warpPerspective` cv2 function and then I did `addWeighted` to overlap the lanes to the background.   

Below is the result of tracked image through this pipeline!!   

![alt text][image8]

---

### Pipeline (video)

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is my final video output.
[LINK](./tracked_project_video.mp4)

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I need to imporve the finding lane pixels' algorithm. I think that the results of the images look good, but the problem appears when I watch the result of the video. Relying on the brightness and curvation degree of the lanes, suddenly, I see the wrong detected lanes. The critical problem of this algorithm is how to find the next left and right start points. Currently, it just calculates the mean values. As I saw the Q&A video, I have to consider the convolution methods or another method to detect next start points well.
   
