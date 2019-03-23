import cv2
import numpy as np


def Canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    # cv2.imshow("blur", blur_image)
    # cv2.imshow("gray", gray_image)
    # cv2.imshow("canny", canny_image)
    return canny_image


def make_coordinates(image, line_parameters):
    # print("line is: ", line_parameters)
    if not np.isnan(line_parameters).all():
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print("left_fit is: ", left_fit)
    # print("right_fit is: ", right_fit)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # print(left_fit_average, right_fit_average)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    # print(lines)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def find_lanes(image):
    # image = cv2.imread("./test_image.jpg")
    lane_image = np.copy(image)
    canny_image = Canny(lane_image)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)
    average_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, average_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # cv2.imshow("origin", lane_image)
    # cv2.imshow("cropped_canny",cropped_canny)
    cv2.imshow("result", combo_image)
    cv2.waitKey(1)


cap = cv2.VideoCapture("./test2.mp4")
while (cap.isOpened()):
    _, frame_image = cap.read()
    find_lanes(frame_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
