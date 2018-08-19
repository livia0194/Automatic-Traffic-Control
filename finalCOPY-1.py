import numpy as np
import cv2
import os
import time
import Signal_GPIO as SG
from StringIO import StringIO
import logging
import logging.handlers
import random
import imutils
from threading import Thread

global time_A,time_B,time_C,time_D
global w,h,signal_change
Signal_Thrushold = 120
#import matplotlib.pyplot as plt
from  scipy import ndimage
import numpy as np

All_Points = []
History = [0,0,0,0]
History_index = 0


class Traffic:
    def  __init__(self):
        self.value1 = 1
        self.value2 = 2
    def get_traffic(self):
        global time_A,time_B,time_C,time_D
        time_A = 5
        time_B = 6
        time_C = 5
        time_D = 6
        
        
    def get_section_cord(self,points,w,h):
        Section_Ax =[]
        Section_Ay =[]
        Section_Bx =[]
        Section_By =[]
        Section_Cx =[]
        Section_Cy =[]
        Section_Dx =[]
        Section_Dy =[]


        Section_Ax.append(0)
        Section_Ax.append(w/2)
        Section_Ay.append(0)
        Section_Ay.append(h/2)
        print (Section_Ax)
        print (Section_Ay)

        Section_Bx.append(w/2)
        Section_Bx.append(w)
        Section_By.append(0)
        Section_By.append(h/2)
        print (Section_Bx)
        print (Section_By)

        Section_Cx.append(w/2)
        Section_Cx.append(w)
        Section_Cy.append(h/2)
        Section_Cy.append(h)
        print Section_Cx
        print Section_Cy

        Section_Dx.append(0)
        Section_Dx.append(w/2)
        Section_Dy.append(h/2)
        Section_Dy.append(h)
        print Section_Dx
        print Section_Dy
        A= []
        B= []
        C= []
        D= []
        print len(points)
##        print ("test result a")
        Row_cntr = 0
        for i in range (len(points)):
           if points[i][0]>=Section_Ax[0] and points[i][0]<=Section_Ax[1]: #checking x range of co-ordinates
               if points[i][1]>=Section_Ay[0] and points[i][1]<=Section_Ay[1]: #checking for y range co-ordinates
                   A.append([])
                   A[Row_cntr].append (points[i][0])
                   A[Row_cntr].append (points[i][1])
                   A[Row_cntr].append (points[i][2])
                   Row_cntr+=1
        Row_cntr =0           
##        print ("test result b")
        for i in range (len(points)):
           if points[i][0]>=Section_Bx[0] and points[i][0]<=Section_Bx[1]: #checking x range of co-ordinates
               if points[i][1]>=Section_By[0] and points[i][1]<=Section_By[1]: #checking for y range co-ordinates
                   B.append([])
                   B[Row_cntr].append (points[i][0])
                   B[Row_cntr].append (points[i][1])
                   B[Row_cntr].append (points[i][2])
                   Row_cntr+=1
        Row_cntr =0           
##        print ("test result c")
        for i in range (len(points)):
           if points[i][0]>=Section_Cx[0] and points[i][0]<=Section_Cx[1]: #checking x range of co-ordinates
               if points[i][1]>=Section_Cy[0] and points[i][1]<=Section_Cy[1]: #checking for y range co-ordinates
                   C.append([])
                   C[Row_cntr].append (points[i][0])
                   C[Row_cntr].append (points[i][1])
                   C[Row_cntr].append (points[i][2])
                   Row_cntr+=1
                   
        Row_cntr =0
##        print ("test result d")
        for i in range (len(points)):
           if points[i][0]>=Section_Dx[0] and points[i][0]<=Section_Dx[1]: #checking x range of co-ordinates
               if points[i][1]>=Section_Dy[0] and points[i][1]<=Section_Dy[1]: #checking for y range co-ordinates
                   D.append([])
                   D[Row_cntr].append (points[i][0])
                   D[Row_cntr].append (points[i][1])
                   D[Row_cntr].append (points[i][2])
                   Row_cntr+=1
            
##    print ("printing result a")
##    print (A)
##    print ("printing result b")
##    print (B)
##    print ("printing result c")
##    print (C)
##    print ("printing result d")
##    print (D)
                                 
       
        return A, B, C, D
    
    def get_Imagesize(self,Image_name):
        imageGray = cv2.cvtColor(Image_name, cv2.COLOR_BGR2GRAY)
        w,h=imageGray.shape[::-1] 
        return w,h
    def Draw_Contour(self,Input_Image, color,Cnt):
        cv2.drawContours(Input_Image,Cnt,-1,color,2)
        return Input_Image
    def Get_Connected_Img(self,Input_Image,img_count,Background):
        Background=cv2.imread("/home/pi/Traffic control/backg/_%d.png"%0)
        major_contour = 0
        center_points=[]
        row_ptr=0
        cp_x=0
        cp_y=0
        ContourArea = [4]
        
        main_img= self.render_image(Input_Image)

        sub = cv2.subtract(Background, main_img)
        cv2.imwrite("/home/pi/Traffic control/subtract/_%d.png"%img_count,sub)
        cv2.imwrite("/home/pi/Traffic control/backg/_%d.png"%img_count,Background)
        
        med = cv2.medianBlur(sub, 9)
        kernel = np.ones((5,5), np.uint8)


        edges = cv2.Canny(med,100,200)

        img_dilation = cv2.dilate(edges, kernel, iterations=2)


    ##    cv2.imwrite('/home/pi/median5.png',img_dilation)

        minLineLength = 100
        maxLineGap = 10

        
        labeled_array, num_features1 = ndimage.measurements.label(img_dilation)
        cars=np.array([num_features1])
        print("cars"+ str(cars))

        contours ,_= cv2.findContours(img_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       
        LENGTH = len(contours)
        print("Contour Len" + str(LENGTH))
        status = np.zeros((LENGTH,1))
        Total_Counts = 0
        print("joining contours")
        for i,cnt1 in enumerate(contours):
            
            x = i    
            if i != LENGTH-1:
                for j,cnt2 in enumerate(contours[i+1:]):
                    x = x+1
                    dist = self.find_if_close(cnt1,cnt2)
                    if dist == True:
                        val = min(status[i],status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x]==status[i]:
                            status[x] = i+1

        unified = []
        color =(255, 0, 0)
##        print("joining contours complete")
##        print(int(status.max()))
        try:
            maximum = int(status.max())+1
##            print ("maximum = " ,maximum)
##            print "printing Error"
            for Cnt_1 in xrange(maximum):
                #print ("printing i", Cnt_1)
                pos = np.where(status==Cnt_1)[0]
                if pos.size != 0 :
                    Total_Counts = Total_Counts+1
                    cont = np.vstack(contours[Cnt_1] for Cnt_1 in pos)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)
                    A=int(cv2.contourArea(hull))
                    x,y,w,h = cv2.boundingRect(hull)
                    cp_x=x+w/2
                    cp_y=y+h/2
                    #print ("printing Area" +str(A))
                    
                    if A >= 500:
                        #print ("printing points " +str(A))
                        center_points.append([])
                        center_points[row_ptr].append(cp_x)
                        center_points[row_ptr].append(cp_y)
                        center_points[row_ptr].append(A)
                        row_ptr+=1
                        col = 'green'
                        color =(0,255,0)
                        major_contour = major_contour + 1

##                        print(str(color))
##                        print('printing Green')
                        
                    else:

                        
                        col = 'Red'
                        color =(255,0,0)
##                        print(str(color))
##                        print('printing Red')
                    self.Draw_Contour(Input_Image,(0,0,255),unified)
                    cv2.rectangle(Input_Image, (x,y), (x+w,y+h), color, 2)
                    #print col

        except:
            print ('Error')
            print(status.max)
        print('Total Counts = ' + str(Total_Counts))

        return Input_Image, major_contour,center_points
        
    def render_image(self,img_RGB):
        img_rgb = img_RGB
        
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in xrange(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        ##upsample image to original size
        for _ in xrange(numDownSamples):
            img_color = cv2.pyrUp(img_color)
            

         # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x,y,z) = img_color.shape
        img_edge = cv2.resize(img_edge,(y,x)) 
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        cv2.waitKey(0)
        blur = cv2.bilateralFilter(img_edge,9,75,75)

        print img_edge.shape, img_color.shape
        cartoon = img_edge #cv2.bitwise_and(img_color, img_edge)
        return cartoon


    def find_if_close(self,cnt1,cnt2):
        row1,row2 = cnt1.shape[0],cnt2.shape[0]
        for i in xrange(row1):
            for j in xrange(row2):
                dist = np.linalg.norm(cnt1[i]-cnt2[ j])
                if abs(dist) < 5 :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False


##start of main flow
                
    def Get_Time_Percent(self, Max_count,Present_count):
        Result_count = (Present_count *100)/Max_count
        return Result_count 
    
    def Get_Background(self):
        traffic_class = Traffic                
        cap = cv2.VideoCapture(0)
        ret,image = cap.read()
        cap.release()
        cv2.imshow('Background',image)
        cv2.waitKey(1)
        w,h= self.get_Imagesize(image)
        print (w,h)


        
        Background = self.render_image(image)
        cv2.imshow('Background',Background)
        cv2.waitKey(1)
        return Background,w,h

    def Run(self,Background,w,h):
        global time_A,time_B,time_C,time_D,signal_change
        Accumulator = 0
        _Continue = True
        while (_Continue):
            for images in range(3):
                cap = cv2.VideoCapture(0)
                ret, Front_image = cap.read()
                cv2.imwrite("/home/pi/Traffic control/image_Store/Captured_%d.png"%images, Front_image)

                cap.release()

            offset  = 0;
            for i in range(3):
                print ('session = '+ str(i))
                
            ##    img=X_data[i]
                img =cv2.imread("/home/pi/Traffic control/image_Store/Captured_%d.png"%i)
                cv2.imshow("test image",img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                front1 , cntr, points  = self.Get_Connected_Img(img,i,Background)
                cv2.imwrite("/home/pi/Traffic control/Result_Store/Result_%d.png"%i, front1)
                if(len(points) > 0):
                    Accumulator += cntr
                    for j in range (len (points)):
                        All_Points.append([])
                        All_Points[offset].append(points[j][0])
                        All_Points[offset].append(points[j][1])
                        All_Points[offset].append(points[j][2])
                        offset+=1
                    print ('Printing points')
                    print(All_Points)
                
                else:
                    print ('No points')
            

            result_a, result_b, result_c, result_d= self.get_section_cord(All_Points,w,h)

            Result_Count = Accumulator / 3
            average_a = len (result_a) / 3
            print (average_a)
            average_b = len (result_b) / 3
            print (average_b)
            average_c = len (result_c) / 3
            print (average_c)
            average_d = len (result_d) / 3
            print (average_d)
            print('Result Count =' + str(Result_Count))
            result_Count = 0
            signal_change = [average_a,average_b,average_c,average_d]
            
            
##            Area_ResultA = 0
##            Area_ResultB = 0
##            Area_ResultC = 0
##            Area_ResultD = 0
##
##            Avg_resultA = 0
##            Avg_resultB = 0
##            Avg_resultC = 0
##            Avg_resultD = 0
##            
##            if len(result_a) > 0:      
##                for i in range(len(result_a)):
##                    Area_ResultA += result_a[i][2]
##                    result_Count +=1
##                Avg_resultA =  Area_ResultA /  result_Count
##            print(Avg_resultA)
##            result_Count = 0
##            
##            if len(result_b) > 0:    
##                for i in range(len(result_b)):
##                    Area_ResultB += result_b[i][2]
##                    result_Count +=1
##                Avg_resultB =  Area_ResultB /  result_Count
##            print(Avg_resultB)
##            result_Count = 0
##            
##            if len(result_c) > 0:     
##                for i in range(len(result_c)):
##                    Area_ResultC += result_c[i][2]
##                    result_Count +=1
##                Avg_resultC =  Area_ResultC /  result_Count
##            print(Avg_resultC)
##            result_Count = 0
##
##            if len(result_d) > 0:
##                for i in range(len(result_d)):
##                    Area_ResultD += result_d[i][2]
##                    result_Count +=1
##                Avg_resultD =  Area_ResultD /  result_Count
##            print(Avg_resultD)
##            result_Count = 0
            loop_counter=4
            least_priority = 0
            final_result=[average_a,average_b,average_c,average_d]
            if History[0]=='X':
                final_result = [average_a,average_b,average_c,average_d]
                
            else :
                Get_pre_result= History[History_index]
                if Get_pre_result== 'a':
                    final_result = [0,average_b,average_c,average_d]
                if Get_pre_result== 'b':
                    final_result = [average_a,0,average_c,average_d]
                if Get_pre_result== 'c':
                    final_result = [average_a,average_b,0,average_d]
                if Get_pre_result== 'd':
                    final_result = [average_a,average_b,average_c,0]
                
            Green_signal= np.argmax(final_result)
            print ('Green Signal: ' + str(Green_signal))
            duplicate_values = ['X', 'X', 'X', 'X']
            lane_array = ['a', 'b', 'c', 'd']
            least_present = False
            dup_values=0
            for dup_values in range(loop_counter):
                if final_result[dup_values]== final_result[Green_signal]:
                    duplicate_values[dup_values]= lane_array[dup_values]
                    if History [dup_values]=='X':
                        least_priority += se 
                         
            if dup_values > 0:
                if (least_present):
                    for i in range (4):
                        if least_priority and 2**i:
                            Green_signal = i
            print("Printing Percentage")
##            print self.Get_Time_Percent(6,average_a)
            time_A = Signal_Thrushold * self.Get_Time_Percent(6,average_a) / 100
            time_B = Signal_Thrushold * self.Get_Time_Percent(6,average_b) / 100
            time_C = Signal_Thrushold * self.Get_Time_Percent(6,average_c) / 100
            time_D = Signal_Thrushold * self.Get_Time_Percent(6,average_d) / 100
##            print ('Printing calculated time')
##            print time_A,time_B,time_C,time_D
                  
                
##            print duplicate_values        
##            History[History_index] =lane_array[Green_signal]
##            History_index+=1
##            print ('History_index:' + str(History_index))
##            if History_index>=4:
##                History_index=0
##            print ('History_index:' + str(History_index))
##            print ('History:' , str(History))
            black=(0,0,0)
            cv2.rectangle(Background, (0,0), (w/2,h/2), black, 2)
            cv2.rectangle(Background, (w/2,0), (w,h/2), black, 2)
            cv2.rectangle(Background, (0,h/2), (w/2,h), black, 2)
            cv2.rectangle(Background, (w/2,h/2), (w,h), black, 2)
            cv2.imwrite("/home/pi/Traffic control/Sectored_img/sectored_%d.png"%images,Background)
            
            del All_Points[:]
            _Continue = False
App_Ref = Traffic()
AppSignals = SG.Signals()
cnt = 0
##Time_A = 5
##Time_B = 5
##Time_C = 5
##Time_D = 5
##Time_A = Time_A *  Signal_Thrushold
##Time_B = Time_B *  Signal_Thrushold
##Time_C = Time_C *  Signal_Thrushold
##Time_D = Time_D *  Signal_Thrushold
time_A = 2
time_B = 2
time_C = 2
time_D = 2
cnt = 0
Back_img,w,h = App_Ref.Get_Background()
AppSGThread = Thread(target = AppSignals.Set_Signal(5))
while(1):
    
    AppRefThread = Thread(target = App_Ref.Run(Back_img,w,h))
    AppRefThread.start()
    Max_index= np.argmax(signal_change)
    cnt= Max_index
    signal_change[Max_index] = 0
    AppSGThread = Thread(target = AppSignals.Set_Signal(cnt))
    print  time_A,time_B,time_C,time_D
    AppSGThread.start()
    time.sleep(time_A)
    
    Max_index= np.argmax(signal_change)
    cnt= Max_index
    signal_change[Max_index] = 0
    AppSGThread = Thread(target = AppSignals.Set_Signal(cnt))
    print  time_A,time_B,time_C,time_D
    AppSGThread.start()
    time.sleep(time_B)

    
    Max_index= np.argmax(signal_change)
    cnt= Max_index
    signal_change[Max_index] = 0
    AppSGThread = Thread(target = AppSignals.Set_Signal(cnt))
    print  time_A,time_B,time_C,time_D
    AppSGThread.start()
    time.sleep(time_C)

    
    Max_index= np.argmax(signal_change)
    cnt= Max_index
    signal_change[Max_index] = 0
    AppSGThread = Thread(target = AppSignals.Set_Signal(cnt))
    print  time_A,time_B,time_C,time_D
    AppSGThread.start()
    time.sleep(time_D)

    
    
    AppSGThread = Thread(target = AppSignals.Set_Signal(5))
    print  time_A,time_B,time_C,time_D
