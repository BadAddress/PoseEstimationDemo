#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // 设置棋盘格尺寸
    Size patternSize(6, 10); // 指定内角点的数量

    // 存储世界坐标点和图像坐标点
    vector<vector<Point3f>> objPoints;
    vector<vector<Point2f>> imgPoints;

    // 构造世界坐标系统中的点
    vector<Point3f> objP;
    for (int i = 0; i < patternSize.height; i++)
    {
        for (int j = 0; j < patternSize.width; j++)
        {
            objP.emplace_back(j, i, 0); // 填充模式
        }
    }

    // 查找目录中的图像文件
    vector<String> imageFiles;
    glob("/home/bl/Desktop/phoneCalib/*.jpg", imageFiles, false);

    // 图像尺寸变量
    Size imageSize;

    // 遍历每个图像文件
    for (const auto& imageFile : imageFiles)
    {
        Mat image = imread(imageFile, IMREAD_GRAYSCALE);
        if (image.empty())
            continue; // 如果图像未正确加载则跳过

        // 图像预处理
        // Mat processedImage;
        // equalizeHist(image, processedImage); // 增强对比度
        // GaussianBlur(processedImage, processedImage, Size(5, 5), 0); // 降噪

        vector<Point2f> corners;
        bool found = findChessboardCorners(image, patternSize, corners,
                                           CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        if (found)
        {
            // 精细化角点位置
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
            cornerSubPix(image, corners, Size(11, 11), Size(-1, -1), criteria);

            // 添加到点集
            objPoints.push_back(objP);
            imgPoints.push_back(corners);

            // drawChessboardCorners(processedImage, patternSize, Mat(corners), found);
            // imshow("Chessboard Corners", processedImage);
            // waitKey(8000); // 等待500毫秒
            // cout<<"###"<<endl;
        }

        // 更新图像尺寸
        imageSize = image.size();
    }

    // 确保有足够的图像用于标定
    if (imgPoints.size() < 1)
    {
        cout << "Not enough images to calibrate" << endl;
        return 1; // 退出程序
    }

    // 标定相机内参数
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    int flags = CALIB_FIX_ASPECT_RATIO + CALIB_FIX_K4 + CALIB_FIX_K5;
    double rms = calibrateCamera(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);

    // 输出结果
    cout << "Camera Matrix: \n" << cameraMatrix << endl;
    cout << "Distortion Coefficients: \n" << distCoeffs << endl;
    cout << "RMS Error: " << rms << endl;

    return 0;
}
