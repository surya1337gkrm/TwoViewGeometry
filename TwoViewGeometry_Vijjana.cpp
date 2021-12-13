#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <fstream>
#include<iostream>
#include<vector>

using namespace cv;
using namespace std;

Mat imgLeft=imread("C:/Sem01_Fall2021/Advanced Computer Vision/TwoViewGeometry/Left_0.bmp");
Mat imgRight = imread("C:/Sem01_Fall2021/Advanced Computer Vision/TwoViewGeometry/Right_0.bmp");
Mat tempL,tempR,pointMat,abcMat, F,lPatch,rPatch;

Rect selectionL,selectionR;

double xp,yp,nccScore,ncc;

vector<double> ndata;
vector<vector<double>> nccData;


//Compute the NCC of two images or patches
//Requirement: img1 and img2 have the same dimension

double getNCC(Mat& img1, Mat& img2)
{
   

    Scalar avg1 = mean(img1);
    Scalar avg2 = mean(img2);


    Mat new_img1 = img1 - avg1[0];
    
    Mat new_img2 = img2 - avg2[0];

    Mat mag1 = new_img1.mul(img1);
    double sum1 = sqrt(sum(mag1)[0]);

    Mat mag2 = new_img2.mul(img2);
    double sum2 = sqrt(sum(mag2)[0]);

    double inner_product = sum(new_img1.mul(new_img2))[0];

    ncc= (inner_product / (sum1 * sum2));
    return ncc;
}



void onMouseClickLeft(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN) {
        tempL.copyTo(imgLeft);
        tempR.copyTo(imgRight);

        if ( x>=1 && y>=1 && x<= tempL.cols-2 && y<= tempL.rows-2) {
           
            selectionL = Rect(x - 1, y-1, 3, 3);
            lPatch = tempL(selectionL);
        }
        
        circle(imgLeft, Point(x, y), 5, Scalar(0, 0, 255),FILLED);   

        pointMat.ptr<double>(0)[0] = x;
        pointMat.ptr<double>(1)[0] = y;
        pointMat.ptr<double>(2)[0] = 1;

        abcMat = F * pointMat;  

        double a = abcMat.ptr<double>(0)[0];
        double b = abcMat.ptr<double>(1)[0];
        double c = abcMat.ptr<double>(2)[0];

        double max = 0;
        double xmax = 0; double ymax = 0;

        for (xp = 1; xp <= imgRight.cols-2; xp++) {
            yp = -((c + (a * xp)) / b);
            
            if (yp >= 1 && yp <= imgRight.rows-2) {

                circle(imgRight, Point(xp, yp), 1, Scalar(0, 0, 255), FILLED);
                   
                    selectionR = Rect(xp - 1, yp-1, 3, 3);
                    rPatch = tempR(selectionR);

                    nccScore = getNCC(lPatch, rPatch);
                    if (nccScore > max) {
                        max = nccScore;
                        xmax = xp;
                        ymax = yp;
                        
                    }   
            }
        }
        
        circle(imgRight, Point(xmax, ymax), 5, Scalar(0, 255, 0), FILLED);
    }
}

int main(int argc, char** argv)
{
    
    namedWindow("camera_left", WINDOW_NORMAL);
    namedWindow("camera_right", WINDOW_NORMAL);
    setMouseCallback("camera_left", onMouseClickLeft, NULL);


    
    imgLeft.copyTo(tempL);
    imgRight.copyTo(tempR);

    pointMat.create(3, 1, CV_64FC1);

    double values1[3][3] = { {877.24128, 0, 543.98552}, {0, 876.55123 , 368.43676 }, {0, 0, 1} };
    Mat K1 = Mat(3, 3, CV_64FC1, values1);


    double values2[3][3] = { {878.47023, 0, 541.75639}, {0, 877.80629, 393.95156}, {0, 0, 1} };
    Mat K2 = Mat(3, 3, CV_64FC1, values2);

    double t[3] = { 294.92002,7.71859, 102.50458 };
    double tx_data[3][3] = { {0, -t[2], t[1]}, {t[2], 0, -t[0]}, {-t[1], t[0], 0} };
    Mat t_x = Mat(3, 3, CV_64FC1, tx_data);

    // step 2: You need to create rotation matrix
    double y_data[3][3] = { {0.7646,0.0616, -0.6416}, {-0.0726,0.9973,0.0092}, {0.6404,0.0396,0.7670} };
    Mat R = Mat(3, 3, CV_64FC1, y_data);

    // step 3: compute the fundamental matrix and print it out
    // K'^(-T) * [t]x * R * K^(-1)
    Mat K2_inv_t, K1_inv;
    K1_inv = K1.inv();
    transpose(K2.inv(), K2_inv_t);
    F = K2_inv_t * t_x * R * K1_inv; //Fundamental matrix
    

    while (1) {
        imshow("camera_left",imgLeft);
        imshow("camera_right", imgRight);
        uchar w = waitKey(1);
        
        if (w == 27) {
            break;
        }
    }
    return 1;
}