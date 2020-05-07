import cv2
import numpy as np
import os

def preprocess_image(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1200,400))
    cv2.imshow('Raw', image)

    blur = cv2.GaussianBlur(image, (5,5), 0)
    cv2.imshow('Gaussian Blur', blur)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_image = gray_image
    cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY, bin_image)
    cv2.imshow('Binary Image', bin_image)

    kernel = np.array([-1, 0, 1])
    img_edge = cv2.filter2D(bin_image, -1, kernel, (-1,-1))
    cv2.imshow('Edge Detection', img_edge)

    mask = np.zeros(img_edge.shape, dtype='uint8')
    pts = np.array([[50,400], [350,230], [717,230], [1000,400]])
    cv2.fillConvexPoly(mask, pts, (255,0,0), 4)
    cv2.imshow('mask', mask)
    img = cv2.bitwise_and(img_edge, mask)
    cv2.imshow('Edge Final', img)
    img = cv2.erode(img, (5,5))
    cv2.imshow('Edge Final1', img)

    lines = cv2.HoughLinesP(img,1 ,np.pi/180, 20,None,20,100)

    slope_thres = 0.3
    right_lanes = []
    left_lanes = []
    try:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = float(y2-y1)/float(x2-x1)
            if(abs(slope)<slope_thres):
                continue
            if(x1>600 and x2>600):
                right_lanes.append(line[0])
            else:
                left_lanes.append(line[0])
    
        for line in right_lanes:
            x1,y1,x2,y2 = line
            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
        for line in left_lanes:
            x1,y1,x2,y2 = line
            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
    
        right_pts = []
        left_pts = []
        for line in right_lanes:
            x1,y1,x2,y2 = line
            right_pts.append([x1,y1])
            right_pts.append([x2,y2])
        
        right_m = 0.0
        right_b = []
        if (len(right_pts) > 0):
            right_line = cv2.fitLine(np.float32(right_pts), cv2.DIST_L2, 0, 0.01, 0.01)
            right_m = right_line[1] / right_line[0]
            right_b = [right_line[2], right_line[3]]
        
        for line in left_lanes:
            x1,y1,x2,y2 = line
            left_pts.append([x1,y1])
            left_pts.append([x2,y2])
        
        left_m = 0.0
        left_b = []
        if (len(left_pts) > 0):
            left_line = cv2.fitLine(np.float32(left_pts), cv2.DIST_L2, 0, 0.01, 0.01)
            left_m = left_line[1] / left_line[0]
            left_b = [left_line[2], left_line[3]]

        ini_y = 230
        fin_y = 400

        right_ini_x = (float(ini_y - right_b[1]) / right_m) + right_b[0];
        right_fin_x = (float(fin_y - right_b[1]) / right_m) + right_b[0];

        left_ini_x = (float(ini_y - left_b[1]) / left_m) + left_b[0];
        left_fin_x = (float(fin_y - left_b[1]) / left_m) + left_b[0];

        output = []
        output.append([left_ini_x, ini_y])
        output.append([right_ini_x, ini_y])
        output.append([right_fin_x, fin_y])
        output.append([left_fin_x, fin_y])
        output=np.int32(output)
        output_image = np.zeros(image.shape)
        cv2.fillConvexPoly(output_image, output, (0,255,0), 0)
        image = np.uint8((0.3*output_image)+(0.7*image))

    except:
        pass

    cv2.imshow('Final Image', image)

    cv2.waitKey(10)

folder = '../datasets/2011_09_26/2011_09_26_drive_0015_sync/image_02/data'
for img in sorted(list(os.listdir(folder)))[6:]:
    preprocess_image(os.path.sep.join([folder, img]))
# preprocess_image('/home/archit/Documents/project_iith/Object_Detection_Training/Object_Detection_Prediction/datasets/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000006.png')
# preprocess_image('../datasets/2011_09_26_drive_0001_sync/image_02/data/0000000009.png')