#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void getFeaturePointsAndMatch(Mat img1,Mat img2,vector<KeyPoint>& kps1,vector<KeyPoint>& kps2,vector<DMatch>& matches);

cv::Point2d pixelToCamNorm(const Point2d& pixelCordinate, const Mat& intrinsicMatrix);

void bundleAdjustmentGaussNewton(
  const VecVector3d& points_3d,
  const VecVector2d& points_2d,
  const Mat& intrinsicMatrix,
  Sophus::SE3d& Transformation);



int main(){
    string imgPath_1 = "../assets/color_1.png";
    string imgPath_2 = "../assets/color_2.png";
    
    cv::Mat img_1 = cv::imread(imgPath_1, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(imgPath_2, cv::IMREAD_COLOR);
    cv::Mat d1 = imread("../assets/depth_1.png", cv::IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像

    cv::Mat K = (cv::Mat_<double>(3, 3) << 911.666,       0, 635.311,
                                                 0, 911.376, 371.241,
                                                 0,       0,       1);

    vector<cv::KeyPoint> kps1,kps2;
    vector<cv::DMatch> matches;
    getFeaturePointsAndMatch(img_1,img_2,kps1,kps2,matches);

    vector<cv::Point3f> pts_1_3d;
    vector<cv::Point2f> pts_2_pixel;

    for (DMatch m:matches) {
        auto depth = d1.at<ushort>( cvRound(kps2[m.queryIdx].pt.y) ,  cvRound(kps1[m.queryIdx].pt.x) ); 
        
        if (depth == 0) continue;

        Point2d p1_camNorm = pixelToCamNorm(kps1[m.queryIdx].pt, K);
        /*
            camera normalized coordinate  =>             [     X_norm       ,       Y_norm       ,    1    ]

            actural camera coordinate     =>             [  X_norm * Depth  ,   Y_norm * Depth   ,  Depth  ]

         >> scaled  camera coordinate     =>     (1/s) * [  X_norm * Depth  ,   Y_norm * Depth   ,  Depth  ]  
        */
        const static float scaling_factor = 5000.0;
        pts_1_3d.push_back(Point3f(p1_camNorm.x*depth/scaling_factor, p1_camNorm.y*depth/scaling_factor, depth/scaling_factor));
        pts_2_pixel.push_back(kps2[m.trainIdx].pt);
    } 

    VecVector3d pts_1_3d_eigen;
    VecVector2d pts_2_pixel_eigen;
    for (size_t i = 0; i < pts_1_3d.size(); ++i) {
        pts_1_3d_eigen.push_back(Eigen::Vector3d(pts_1_3d[i].x, pts_1_3d[i].y, pts_1_3d[i].z));
        pts_2_pixel_eigen.push_back(Eigen::Vector2d(pts_2_pixel[i].x, pts_2_pixel[i].y));
    }


    Sophus::SE3d Trans;
    bundleAdjustmentGaussNewton(pts_1_3d_eigen,pts_2_pixel_eigen,K,Trans);
    return 0;
}





cv::Point2d pixelToCamNorm(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


void bundleAdjustmentGaussNewton(
  const VecVector3d& points_3d,
  const VecVector2d& points_2d,
  const Mat& K,
  Sophus::SE3d& Trans)
{   typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterationCnt = 10;
    double cost = 0,lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for(int iter=0; iter<iterationCnt; ++iter){
        Eigen::Matrix<double,6,6> H =  Eigen::Matrix<double,6,6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost =0;
         
        for(int i=0; i<points_3d.size();++i){
            Eigen::Vector3d P_hat = Trans * points_3d[i];

            double inv_Z_hat = 1 / P_hat[2];
            double inv_Z_hat_Sq = pow(inv_Z_hat,2);
            
            Eigen::Vector2d points_2d_hat(fx * P_hat[0] * inv_Z_hat + cx , 
                                          fy * P_hat[1] * inv_Z_hat + cy);
            
            //Residual Block  ====>   Reprojection Error
            Eigen::Vector2d err = points_2d[i] - points_2d_hat;
            cost += err.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;

            J << 
                -fx * inv_Z_hat,
                0,
                fx * P_hat[0] * inv_Z_hat_Sq,
                fx * P_hat[0] * P_hat[1] * inv_Z_hat_Sq,
                -fx - fx * pow(P_hat[0],2) * inv_Z_hat_Sq,
                fx * P_hat[1] * inv_Z_hat,

                0,
                -fy * inv_Z_hat,
                fy * P_hat[1] * inv_Z_hat_Sq,
                fy + fy * pow(P_hat[1],2) * inv_Z_hat_Sq,
                -fy * P_hat[0] * P_hat[1] * inv_Z_hat_Sq,
                -fy * P_hat[0] * inv_Z_hat;

            H = J.transpose() * J;
            b += -J.transpose() * err;
        }

        Vector6d deltaKesai; 
        deltaKesai = H.ldlt().solve(b);
        
        if (isnan(deltaKesai[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        } 
        
        Trans = Sophus::SE3d::exp(deltaKesai) * Trans;        
        
        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (deltaKesai.norm() < 1e-6) {
            // converge
            break;
        }

    }
    cout << "pose by g-n: \n" << Trans.matrix() << endl;
}


void getFeaturePointsAndMatch(Mat img1,Mat img2,vector<KeyPoint>& kps1,vector<KeyPoint>& kps2,vector<DMatch>& matches){
    Ptr<ORB> orb = ORB::create();
    
    // 用于存放两幅图像的特征描述符
    Mat descriptors1, descriptors2;
    
    // 检测特征点并计算描述符
    orb->detectAndCompute(img1, Mat(), kps1, descriptors1);
    orb->detectAndCompute(img2, Mat(), kps2, descriptors2);
    
    // 初始化BFMatcher，使用默认的L2距离作为距离测量
    BFMatcher matcher(NORM_HAMMING);
    
    // 进行匹配
    matcher.match(descriptors1, descriptors2, matches);
    vector<DMatch> matches_new;
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            matches_new.push_back(matches[i]);
        }
    }
    matches = matches_new;
}