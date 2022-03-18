import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import numpy as np



def grey(image):
      #convert to grayscale
    image = np.asarray(image)
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

  #Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

  #outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges



def region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])
    
def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


### DETECTING LANE LINES IN A VIDEO 

# PATH TO THE VIDE0
video = r"testvideo.mp4"
cap = cv2.VideoCapture(video)
print("here")
num = 0
while(cap.isOpened()): 
    ret, frame = cap.read()
    print(num)
    num += 1
    if ret == True:
        gaus = gauss(frame)
        edges = cv2.Canny(gaus,50,150)
        isolated = region(edges)
        #region of interest, bin size (P, theta), min intersections needed, placeholder array, 
        lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average(frame, lines)
        black_lines = display_lines(frame, averaged_lines)
        #taking wighted sum of original image and lane lines image
        lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
        cv2.imshow("frame", lanes)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break
    else:
        break
cap.release() 
cv2.destroyAllWindows()  
