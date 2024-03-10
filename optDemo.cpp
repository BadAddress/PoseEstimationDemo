#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>          
#include <thread>         
#include <vector>          
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <vector>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


void getMatch(Mat img1,Mat img2,vector<KeyPoint>& kps1,vector<KeyPoint>& kps2,vector<DMatch>& matches){
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



void poseEstimation2d2d(const vector<KeyPoint>& kps1,
                        const vector<KeyPoint>& kps2,
                        const vector<DMatch>& matches,
                        Mat& R,
                        Mat& t){
    
    Mat K = (Mat_<double>(3,3) << 1057.586178106818,                 0, 620.7734472635326,
                                                  0, 1057.586178106818, 855.2250147391001,
                                                  0,                 0,                 1);
    
      //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(kps1[matches[i].queryIdx].pt);
        points2.push_back(kps2[matches[i].trainIdx].pt);
    }

    // //-- 计算基础矩阵
    // Mat fundamental_matrix;
    // fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    // cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    Point2d principal_point(620.4, 855.1);  //相机光心, TUM dataset标定值
    double focal_length = 1057.;      
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;


}


void DrawGrid(double grid_size, int num_grid, int plane, double r, double g, double b); 
void DrawCamera(const Eigen::Isometry3d& T, float lineWidth, float size,bool);


int main(){    

    Mat img_1 = imread("/home/bl/Desktop/PoseEstimation/1.jpg", cv::IMREAD_COLOR);
    Mat img_2 = imread("/home/bl/Desktop/PoseEstimation/2.jpg", cv::IMREAD_COLOR);

    vector<KeyPoint> kps1,kps2;
    vector<DMatch> matches;

    getMatch(img_1,img_2,kps1,kps2,matches);
    
    // Mat img_match;
    // drawMatches(img_1, kps1, img_2, kps2, matches, img_match);
    // imshow("good matches", img_match);
    // waitKey(0);
    
    Mat R,t;
    poseEstimation2d2d(kps1,kps2,matches,R,t);


    Eigen::Isometry3d C0_POSE__W_based = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d R_W_to_C0;
    R_W_to_C0 << 1, 0, 0,
                 0, 0, -1,
                 0, 1, 0;
    Eigen::Vector3d C0_translation = Eigen::Vector3d(3,3,3);
    C0_POSE__W_based.rotate(R_W_to_C0.inverse());
    C0_POSE__W_based.translation() = C0_translation;


    Eigen::Matrix3d R_C0_to_C1;
    Eigen::Vector3d t_C0_to_C1;
    R_C0_to_C1<<R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
                R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
                R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);

    t_C0_to_C1<<t.at<double>(0),t.at<double>(1),t.at<double>(2);

    
    Eigen::Isometry3d C1_POSE__C0_based = Eigen::Isometry3d::Identity();
    C1_POSE__C0_based.rotate(R_C0_to_C1.inverse());
    C1_POSE__C0_based.translation() = -R_C0_to_C1.inverse()*t_C0_to_C1;


    Eigen::Isometry3d C1_POSE__W_based = C0_POSE__W_based * C1_POSE__C0_based ;

    
    


    pangolin::CreateWindowAndBind("World Coordinate Frames", 1024, 720);
    glEnable(GL_DEPTH_TEST);

    // 定义观察相机的投影和初始模型视图矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 500, 500, 320, 240, 0.1, 100),
        pangolin::ModelViewLookAt(4, -4, 3, 0, 0, 0, pangolin::AxisZ)
    );

    // 创建一个交互式的相机视图
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
        .SetHandler(&handler);



    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // 激活相机视图
        d_cam.Activate(s_cam);

        // 绘制坐标系
        glLineWidth(4.0); // 设置线宽为3.0，您可以根据需要调整这个值
        pangolin::glDrawAxis(10.0);
        glLineWidth(1.0); // 设置线宽为3.0，您可以根据需要调整这个值
            // 使用预定义的宏颜色绘制三个方向上的网格
        DrawGrid(0.8, 12, 0, 0.3, 0.3, 0.3); // XY平面
        DrawGrid(0.8, 12, 1, 0.3, 0.3, 0.3); // XZ平面
        DrawGrid(0.8, 12, 2, 0.3, 0.3, 0.3); // YZ平面

        
        
        DrawCamera(C0_POSE__W_based, 4.0, 4,1);
        DrawCamera(C1_POSE__W_based, 4.0, 4,0);
        pangolin::FinishFrame();
    }        

    return 0;
}







void DrawGrid(double grid_size, int num_grid, int plane, double r, double g, double b) {
        glColor3f(r, g, b);
        glBegin(GL_LINES);
        for (int i = -num_grid; i <= num_grid; ++i) {
            double position = grid_size * i;
            if (plane == 0) { // XY plane
                glVertex3f(position, -num_grid * grid_size, 0);
                glVertex3f(position, num_grid * grid_size, 0);
                glVertex3f(-num_grid * grid_size, position, 0);
                glVertex3f(num_grid * grid_size, position, 0);
            } else if (plane == 1) { // XZ plane
                glVertex3f(position, 0, -num_grid * grid_size);
                glVertex3f(position, 0, num_grid * grid_size);
                glVertex3f(-num_grid * grid_size, 0, position);
                glVertex3f(num_grid * grid_size, 0, position);
            } else if (plane == 2) { // YZ plane
                glVertex3f(0, position, -num_grid * grid_size);
                glVertex3f(0, position, num_grid * grid_size);
                glVertex3f(0, -num_grid * grid_size, position);
                glVertex3f(0, num_grid * grid_size, position);
            }
        }
        glEnd();
}




void DrawCamera(const Eigen::Isometry3d& T, float lineWidth, float size,bool isDashed) {
    glPushMatrix();
    glMultMatrixd(T.data());
    glLineWidth(lineWidth);

    if (isDashed) {
        // 启用stipple模式
        glEnable(GL_LINE_STIPPLE);
        // 设置虚线模式 (0x00FF表示将绘制一个短线段,然后跳过一个短线段)
        glLineStipple(1, 0x00FF);
    }

    glBegin(GL_LINES);
    // 绘制相机坐标系的XYZ轴
    glColor3f(1.0, 0.0, 0.0); // X轴为红色
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(size, 0.0, 0.0);
    glColor3f(0.0, 1.0, 0.0); // Y轴为绿色
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, size, 0.0);
    glColor3f(0.0, 0.0, 1.0); // Z轴为蓝色
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, size);
    glEnd();

    if (isDashed) {
        // 禁用stipple模式
        glDisable(GL_LINE_STIPPLE);
    }
    glPopMatrix();
}