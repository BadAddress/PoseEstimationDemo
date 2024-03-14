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

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;




void getFeaturePointsAndMatch(Mat img1,Mat img2,vector<KeyPoint>& kps1,vector<KeyPoint>& kps2,vector<DMatch>& matches);

cv::Point2d pixelToCamNorm(const Point2d& pixelCoordinate, const Mat& intrinsicMatrix);

void bundleAdjustmentGaussNewton(
  const VecVector3d& points_3d,
  const VecVector2d& points_2d,
  const Mat& intrinsicMatrix,
  Sophus::SE3d& Transformation);

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &Transformation);


void DrawGrid(double grid_size, int num_grid, int plane, double r, double g, double b); 
void DrawCamera(const Eigen::Isometry3d& T, float lineWidth, float size,bool);



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
    // bundleAdjustmentGaussNewton(pts_1_3d_eigen,pts_2_pixel_eigen,K,Trans);
    bundleAdjustmentG2O(pts_1_3d_eigen,pts_2_pixel_eigen,K,Trans);


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

            H += J.transpose() * J;
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
/* ========================================================================================================================= */
/* ========================================================================================================================= */
  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> deltaKesai;
    deltaKesai<< update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(deltaKesai) * _estimate;
  }
/* ========================================================================================================================= */
/* ========================================================================================================================= */
  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};







class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTrans> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

/* ========================================================================================================================= */
/* ========================================================================================================================= */
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexTrans* ptr_VertexTrans = static_cast<VertexTrans *> (_vertices[0]);
    Sophus::SE3d Trans_est = ptr_VertexTrans->estimate();
    Eigen::Vector3d pos_pixel = _K * (Trans_est * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexTrans* ptr_VertexTrans = static_cast<VertexTrans *> (_vertices[0]);
    Sophus::SE3d Trans_est = ptr_VertexTrans->estimate();
    Eigen::Vector3d pos_cam = Trans_est * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }
/* ========================================================================================================================= */
/* ========================================================================================================================= */

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};






void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexTrans *vertex_trans = new VertexTrans(); // camera vertex_pose
  vertex_trans->setId(0);
  vertex_trans->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_trans);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, vertex_trans); // set the ith Vertex The edge connected.
    edge->setMeasurement(p2d);  // groundTruth 
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_trans->estimate().matrix() << endl;
  pose = vertex_trans->estimate();
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