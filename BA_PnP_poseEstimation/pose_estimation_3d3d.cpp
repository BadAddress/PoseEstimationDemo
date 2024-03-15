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


using namespace std;
using namespace cv;



void getFeaturePointsAndMatch(Mat img1,Mat img2,vector<KeyPoint>& kps1,vector<KeyPoint>& kps2,vector<DMatch>& matches);

cv::Point2d pixelToCamNorm(const Point2d& pixelCoordinate, const Mat& intrinsicMatrix);

void DrawGrid(double grid_size, int num_grid, int plane, double r, double g, double b); 
void DrawCamera(const Eigen::Isometry3d& T, float lineWidth, float size,bool);

void ICP_SVD(const vector<Point3f>& pts1,const vector<Point3f>& pts2, Mat& Trans_R, Mat& Trans_t); 
void G2O_Trans_Opt(const vector<Point3f>& pts1,const vector<Point3f>& pts2, Mat& Trans_R, Mat& Trans_t);



int main(){
    string imgPath_1 = "../assets/color_1.png";
    string imgPath_2 = "../assets/color_2.png";
    
    cv::Mat img_1 = cv::imread(imgPath_1, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(imgPath_2, cv::IMREAD_COLOR);
    cv::Mat d1 = imread("../assets/depth_1.png", cv::IMREAD_UNCHANGED);
    cv::Mat d2 = imread("../assets/depth_2.png", cv::IMREAD_UNCHANGED);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 911.666,       0, 635.311,
                                                 0, 911.376, 371.241,
                                                 0,       0,       1);

    vector<cv::KeyPoint> kps1,kps2;
    vector<cv::DMatch> matches;
    getFeaturePointsAndMatch(img_1,img_2,kps1,kps2,matches);

    vector<cv::Point3f> pts_1_3d;
    vector<cv::Point3f> pts_2_3d;

    for (DMatch m:matches) {
        auto depth_1 = d1.at<ushort>( cvRound(kps1[m.queryIdx].pt.y) ,  cvRound(kps1[m.queryIdx].pt.x) ); 
        auto depth_2 = d2.at<ushort>( cvRound(kps2[m.trainIdx].pt.y) ,  cvRound(kps2[m.trainIdx].pt.x) ); 
        if (depth_1 == 0 || depth_2==0) continue;

        Point2d p1_camNorm = pixelToCamNorm(kps1[m.queryIdx].pt, K);
        Point2d p2_camNorm = pixelToCamNorm(kps2[m.trainIdx].pt, K);
        /*
            camera normalized coordinate  =>             [     X_norm       ,       Y_norm       ,    1    ]

            actural camera coordinate     =>             [  X_norm * Depth  ,   Y_norm * Depth   ,  Depth  ]

         >> scaled  camera coordinate     =>     (1/s) * [  X_norm * Depth  ,   Y_norm * Depth   ,  Depth  ]  
        */
        const static float scaling_factor = 5000.0;
        pts_1_3d.push_back(Point3f(p1_camNorm.x*depth_1/scaling_factor, p1_camNorm.y*depth_1/scaling_factor, depth_1/scaling_factor));
        pts_2_3d.push_back(Point3f(p2_camNorm.x*depth_2/scaling_factor, p2_camNorm.y*depth_2/scaling_factor, depth_2/scaling_factor));

    } 

    Sophus::SE3d Trans;
    cv::Mat R, t;
    ICP_SVD(pts_1_3d,pts_2_3d,R,t);
    


/*========= VISUALIZE PART ===============================================================================================================================*/
/*========= VISUALIZE PART ===============================================================================================================================*/
/*========= VISUALIZE PART ===============================================================================================================================*/
/*========= VISUALIZE PART ===============================================================================================================================*/

    Eigen::Isometry3d C0_POSE__W_based = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d R_W_to_C0;
    R_W_to_C0 << 1, 0,  0,
                 0, 0, -1,
                 0, 1,  0;
    Eigen::Vector3d C0_translation = Eigen::Vector3d(3,3,3);
    C0_POSE__W_based.rotate(R_W_to_C0.inverse());
    C0_POSE__W_based.translation() = C0_translation;

    
    Eigen::Matrix3d R_C0_to_C1 = Trans.so3().matrix();
    Eigen::Vector3d t_C0_to_C1 = Trans.translation();
    cout<< R_C0_to_C1<<endl;
    cout<< t_C0_to_C1<<endl;

    const static float viz_scaling_factor = 500;
    Eigen::Isometry3d C1_POSE__C0_based = Eigen::Isometry3d::Identity();
    C1_POSE__C0_based.rotate(R_C0_to_C1.inverse());
    C1_POSE__C0_based.translation() = -R_C0_to_C1.inverse()*t_C0_to_C1* viz_scaling_factor; 


    Eigen::Isometry3d C1_POSE__W_based = C0_POSE__W_based * C1_POSE__C0_based ;



    pangolin::CreateWindowAndBind("World Coordinate Frames", 1024, 720);
    glEnable(GL_DEPTH_TEST);

    // 定义观察相机的投影和初始模型视图矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 500, 500, 320, 240, 0.1, 50),
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
}




cv::Point2d pixelToCamNorm(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
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


void ICP_SVD(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W=" << W << endl;

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {
    R_ = -R_;
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
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





/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/
/*==    G2O impl   =============================================================================================================================================================================*/


/// vertex and edges used in g2o ba
class VertexTrans : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};



/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexTrans> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexTrans *pose = static_cast<const VertexTrans *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexTrans* ptr_trans = static_cast<VertexTrans *>(_vertices[0]);
    Sophus::SE3d Trans_est = ptr_trans->estimate();
    Eigen::Vector3d xyz_trans = Trans_est * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) {}

  bool write(ostream &out) const {}

protected:
  Eigen::Vector3d _point;
};