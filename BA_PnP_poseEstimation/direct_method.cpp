#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>
#include <pangolin/pangolin.h>
#include <opencv2/imgproc.hpp>
#include <random>
#include <cmath>


using namespace std;
using namespace cv;



typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> ArrVec_2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> ArrVec_3d;




void DirectPoseEstimation(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const ArrVec_2d& px_ref,
    const vector<double>& depth_ref,
    Sophus::SE3d& Trans_CoordSys1_to_CoordSys2,
    int ScalingDownFactor,
    cv::Mat& CameraInrinsics
);

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


int main(){
    string imgPath_1 = "../assets/color_1.png";
    string imgPath_2 = "../assets/color_2.png";
    
    cv::Mat img_1 = cv::imread(imgPath_1, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(imgPath_2, cv::IMREAD_COLOR);
    cv::Mat d1 = imread("../assets/depth_1.png", cv::IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像

    cv::Mat K = (cv::Mat_<double>(3, 3) << 911.666,       0, 635.311,
                                                 0, 911.376, 371.241,
                                                 0,       0,       1);
    cv::RNG rng;
    cv::Mat gray_1_;
    cv::cvtColor(img_1, gray_1_, cv::COLOR_BGR2GRAY);    
    cv::Mat gray_2_;
    cv::cvtColor(img_2, gray_2_, cv::COLOR_BGR2GRAY);


    cv::Mat gray_1;
    cv::equalizeHist(gray_1_, gray_1);

    cv::Mat gray_2;
    cv::equalizeHist(gray_2_, gray_2);


    int cnt = 2000;
    int boarder = 20;
    
    ArrVec_2d pixels_ref;
    vector<double> depths_ref;
    
    // for (int i = 0; i < cnt; i++) {
    //     int x = rng.uniform(boarder, gray_1.cols - boarder);  // don't pick pixels close to boarder
    //     int y = rng.uniform(boarder, gray_1.rows - boarder);  // don't pick pixels close to boarder
    //     int disparity = d1.at<uchar>(y, x);
    //     if(disparity==0) continue;
    //     double depth = disparity/5000.0; // you know this is disparity to depth
    //     depths_ref.push_back(depth);
    //     pixels_ref.push_back(Eigen::Vector2d(x, y));
    // }



    std::vector<cv::Point2f> corners; // 用于存储检测到的角点
    int maxCorners = cnt; // 最大角点数量
    double qualityLevel = 0.01; // 角点检测的质量水平参数
    double minDistance = 10; // 角点之间的最小距离
    int blockSize = 3; // 计算导数自相关矩阵时使用的块大小
    bool useHarrisDetector = false; // 是否使用Harris角点检测，如果为false，则使用Shi-Tomasi方法
    double k = 0.04; // Harris角点检测方程中的自由参数

    // 使用goodFeaturesToTrack检测角点
    cv::goodFeaturesToTrack(gray_1, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

    for (size_t i = 0; i < corners.size(); i++) {
        int x = static_cast<int>(corners[i].x);
        int y = static_cast<int>(corners[i].y);

        // 确保角点不靠近边界
        if(x < boarder || x >= gray_1.cols - boarder || y < boarder || y >= gray_1.rows - boarder) continue;

        int disparity = d1.at<uchar>(y, x);
        if(disparity == 0) continue; // 忽略disparity为0的点
        double depth = disparity / 50.0; // disparity到depth的转换

        depths_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }









    Sophus::SE3d T_Pre_to_Cur;

    //coarse- to-fine
    for(int ScalingDownFactor = 8; ScalingDownFactor >=1;ScalingDownFactor/=2){
        DirectPoseEstimation(gray_1,gray_2,pixels_ref,depths_ref,T_Pre_to_Cur,ScalingDownFactor,K);
    }

    return 0;
}



void DirectPoseEstimation(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const ArrVec_2d& px_ref,
    const vector<double>& depth_ref,
    Sophus::SE3d& Trans_CoordSys1_to_CoordSys2,
    int ScalingDownFactor,
    cv::Mat& CameraIntrinsics
){
    double fx = CameraIntrinsics.at<double>(0,0)/ScalingDownFactor;
    double fy = CameraIntrinsics.at<double>(1,1)/ScalingDownFactor;
    double cx = CameraIntrinsics.at<double>(0, 2)/ScalingDownFactor;
    double cy = CameraIntrinsics.at<double>(1, 2)/ScalingDownFactor;

    cv::Mat img1_,img2_;
    cv::resize(img1,img1_,cv::Size((int)(img1.cols/ScalingDownFactor),(int)(img1.rows/ScalingDownFactor)));
    cv::resize(img2,img2_,cv::Size((int)(img2.cols/ScalingDownFactor),(int)(img2.rows/ScalingDownFactor)));

    ArrVec_2d px_ref_;
    for(auto px: px_ref){
        px_ref_.push_back(px/ScalingDownFactor);
    }
    
    static const int patchRadius = 1;
    static const int iterations = 100;

    double lastCost = INT_MAX;
    for(int iter=0;iter<iterations;iter++){

        double cost = 0;
        int cntGood = 0;
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        for(int i=0;i<px_ref_.size();i++){

            
            double X_norm = (px_ref_[i][0]-cx)/fx;
            double Y_norm = (px_ref_[i][1]-cy)/fy;

            Eigen::Vector3d P_pre_based ( X_norm*depth_ref[i] , Y_norm*depth_ref[i] , depth_ref[i] );
            Eigen::Vector3d P_cur_based = Trans_CoordSys1_to_CoordSys2 * P_pre_based;
            if(P_cur_based[2]<0) continue;
            
            double u = fx * P_cur_based[0]/P_cur_based[2] + cx ;
            double v = fy * P_cur_based[1]/P_cur_based[2] + cy ;

            if( u-patchRadius<0 ||
                u+patchRadius>img2_.cols ||
                v-patchRadius<0 ||
                v+patchRadius>img2_.rows) continue;


            double X_hat = P_cur_based[0], Y_hat = P_cur_based[1], Z_hat = P_cur_based[2];
            double inv_Z_hat = 1 / P_cur_based[2];
            double inv_Z_hat_Sq = pow(inv_Z_hat,2);
            cntGood++;

            for(int x = -patchRadius; x<=patchRadius; x++){
                for(int y = -patchRadius; y<=patchRadius; y++){
                    double error = GetPixelValue(img1, px_ref_[i][0] + x, px_ref_[i][1] + y) - GetPixelValue(img2, u + x, v + y);
                    
                    Eigen::Matrix<double,2,6> J_trans;
                    Eigen::Vector2d J_gray;


                    J_trans(0, 0) = fx * inv_Z_hat;
                    J_trans(0, 1) = 0;
                    J_trans(0, 2) = -fx * X_hat * inv_Z_hat_Sq;
                    J_trans(0, 3) = -fx * X_hat * Y_hat * inv_Z_hat_Sq;
                    J_trans(0, 4) = fx + fx * X_hat * X_hat * inv_Z_hat_Sq;
                    J_trans(0, 5) = -fx * Y_hat * inv_Z_hat;

                    J_trans(1, 0) = 0;
                    J_trans(1, 1) = fy * inv_Z_hat;
                    J_trans(1, 2) = -fy * Y_hat * inv_Z_hat_Sq;
                    J_trans(1, 3) = -fy - fy * Y_hat * Y_hat * inv_Z_hat_Sq;
                    J_trans(1, 4) = fy * X_hat * Y_hat * inv_Z_hat_Sq;
                    J_trans(1, 5) = fy * X_hat * inv_Z_hat;

                    J_gray = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                    Eigen::Matrix<double,1,6> J;

                    J = -1 * J_gray.transpose() * J_trans;

                    H += J.transpose()*J;
                    b += -J.transpose() * error;
                    cost += error*error;
                }
            }
        }

        Eigen::Matrix<double,6,1> deltaKesai;
        deltaKesai = H.ldlt().solve(b);
            
        if (isnan(deltaKesai[0])) {
            std::cout << "result is nan!" << endl;
           break;
        }                
        if (deltaKesai.norm() < 1e-6) {
            // converge
            break;   
        }
        cost = cost/cntGood;
        if(cost>lastCost){
            break;
        }
        Trans_CoordSys1_to_CoordSys2 = Sophus::SE3d::exp(deltaKesai) * Trans_CoordSys1_to_CoordSys2;        
        std::cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        lastCost = cost;
    }


    cv::Mat img_combined(std::max(img1_.rows, img2_.rows), img1_.cols + img2_.cols, img1_.type());
    img1_.copyTo(img_combined(cv::Rect(0, 0, img1_.cols, img1_.rows)));
    img2_.copyTo(img_combined(cv::Rect(img1_.cols, 0, img2_.cols, img2_.rows)));


    // cv::Mat img1_,img2_;
    for (size_t i = 0; i < px_ref.size(); i+=20) {

        

        double u_pre = px_ref_[i][0];
        double v_pre = px_ref_[i][1];
        double X_norm = (u_pre-cx)/fx;
        double Y_norm = (v_pre-cy)/fy;
        Eigen::Vector3d P_pre_based ( X_norm*depth_ref[i] , Y_norm*depth_ref[i] , depth_ref[i] );
        Eigen::Vector3d P_cur_based = Trans_CoordSys1_to_CoordSys2 * P_pre_based;
        if(P_cur_based[2]<0) continue;

        double u = fx * P_cur_based[0]/P_cur_based[2] + cx ;
        double v = fy * P_cur_based[1]/P_cur_based[2] + cy ;



        cv::circle(img_combined, cv::Point(u_pre, v_pre), 3, cv::Scalar(0, 255, 0), -1);
    
        // 在img2_上绘制点，并加上img1_.cols偏移
        cv::circle(img_combined, cv::Point(u + img1_.cols, v), 3, cv::Scalar(0, 0, 255), -1);
        
        // 绘制连接两点的线，并加上img1_.cols偏移
        cv::line(img_combined, cv::Point(u_pre, v_pre), cv::Point(u + img1_.cols, v), cv::Scalar(255, 0, 0));

    }

    cv::imshow("Combined Image", img_combined);
    cv::waitKey(0);

}


