import cv2 as cv
import numpy as nmp


# region of interest function for yellow lines
def range_of_interest(i_frame):
    cvt = cv.COLOR_BGR2HSV
    cvt2_hsv = cv.cvtColor(i_frame, cvt)
    l_yellow = nmp.array([17, 93, 139])
    u_yellow = nmp.array([49, 255, 255])
    mask = cv.inRange(cvt2_hsv, l_yellow, u_yellow)

    return mask



# applying Gausianblur to reduce noise
def blur(i_frame):
    img1 = range_of_interest(i_frame)

    img_blur = cv.GaussianBlur(img1, (5, 5), 0)


    return img_blur


# draw lines
def draw_line(i_img, i_lines):
    b_image = nmp.zeros((i_img.shape[0], i_img.shape[1], 3), dtype=nmp.uint8)
    if i_lines is not None:
        for l in i_lines:
            for X1, Y1, X2, Y2 in l:
                cv.line(b_image, (X1, Y1), (X2, Y2), (0, 255, 0), thickness=3)
    image = cv.addWeighted(i_img, 0.8, b_image, 1, 0.0)
    return image


# applying canny edge detection and Hough line probabilistic method
def final_process(i_img):
    out_blur = blur(i_img)
    out_canny = cv.Canny(out_blur, 50, 150)

    h_p_lines = cv.HoughLinesP(out_canny, 1, nmp.pi / 180, 50, maxLineGap=150)
    frame1 = draw_line(i_img, h_p_lines)
    return frame1
