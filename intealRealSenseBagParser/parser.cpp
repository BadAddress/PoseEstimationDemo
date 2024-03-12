#include <librealsense2/rs.hpp> // 包含RealSense头文件
#include <opencv2/opencv.hpp> // 包含OpenCV头文件
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


int dev_info() {
    try {
        rs2::config cfg;
        // 指定从.bag文件读取
        cfg.enable_device_from_file("/home/bl/Documents/20240312_043730.bag");
        cfg.enable_stream(RS2_STREAM_DEPTH); // 启用深度流
        cfg.enable_stream(RS2_STREAM_COLOR); // 启用彩色流

        rs2::pipeline p;
        // 启动管道，配置从.bag文件读取
        rs2::pipeline_profile profile = p.start(cfg);

        // 创建对齐对象，将深度帧对齐到彩色帧
        rs2::align align(RS2_STREAM_COLOR);

        while (true) {
            // 等待一组帧，并对它们进行对齐
            auto frames = p.wait_for_frames();
            auto aligned_frames = align.process(frames);

            // 获取对齐后的深度帧和彩色帧
            auto aligned_depth_frame = aligned_frames.get_depth_frame();
            auto color_frame = aligned_frames.get_color_frame();

            if (!aligned_depth_frame || !color_frame) {
                continue; // 如果帧不完整，跳过当前循环
            }

            // 获取对齐后的深度帧的内参（彩色相机的视角）
            auto stream = color_frame.get_profile().as<rs2::video_stream_profile>();
            rs2_intrinsics intrinsics = stream.get_intrinsics();

            // 打印内参
            std::cout << "Aligned Depth (Color Camera View) Intrinsics: " << std::endl;
            std::cout << "Width: " << intrinsics.width << ", Height: " << intrinsics.height << std::endl;
            std::cout << "PPX: " << intrinsics.ppx << ", PPY: " << intrinsics.ppy << std::endl;
            std::cout << "FX: " << intrinsics.fx << ", FY: " << intrinsics.fy << std::endl;
            std::cout << "Distortion Model: " << intrinsics.model << std::endl;

            break; // 读取第一组帧后退出循环，或根据需要进行更多处理
        }
    } catch (const rs2::error & e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}










double calculate_focus_measure(const cv::Mat& image) {
    cv::Mat gray, laplacian;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mu, sigma;
    cv::meanStdDev(laplacian, mu, sigma);
    
    double focusMeasure = sigma.val[0] * sigma.val[0];
    return focusMeasure;
}

void save_frames(const rs2::frame& color_frame, const rs2::frame& depth_frame, int frame_number) {
    // 将深度和彩色帧转换为OpenCV矩阵
    auto color_mat = cv::Mat(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(color_mat, color_mat, cv::COLOR_BGR2RGB); // 转换颜色空间为RGB

    double K = calculate_focus_measure(color_mat);
    if(K<550) return;
    cout<<K<<endl;
    auto depth_mat = cv::Mat(cv::Size(1280, 720), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

    // 为保存的文件命名
    std::string color_file = "./res/color_" + std::to_string(frame_number) + ".png";
    std::string depth_file = "./res/depth_" + std::to_string(frame_number) + ".png";

    // 保存彩色图像和原始深度图像
    cv::imwrite(color_file, color_mat);
    cv::imwrite(depth_file, depth_mat);

    // 将深度数据转换为彩色热力图
    cv::Mat depth_colormap;
    double min;
    double max;
    cv::minMaxIdx(depth_mat, &min, &max);
    cv::convertScaleAbs(depth_mat, depth_colormap, 255 / max);
    cv::applyColorMap(depth_colormap, depth_colormap, cv::COLORMAP_JET);

    // 保存热力图
    std::string heatmap_file = "./res/heat_" + std::to_string(frame_number) + ".png";
    cv::imwrite(heatmap_file, depth_colormap);
    std::cout << "Saved frame " << frame_number << std::endl;
}

int main() try {
    dev_info();
    return 0;
    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_device_from_file("/home/bl/Documents/20240312_043730.bag");
    p.start(cfg);

    rs2::align align_to(RS2_STREAM_COLOR);
    int frame_number = 0;

    while (true) {
        auto frames = p.wait_for_frames();
        auto aligned_frames = align_to.process(frames);
        auto depth_frame = aligned_frames.get_depth_frame();
        auto color_frame = aligned_frames.get_color_frame();
        if (!depth_frame || !color_frame) {
            continue;
        }
        save_frames(color_frame, depth_frame, frame_number);
        frame_number++;

    }

    return EXIT_SUCCESS;
}


catch (const rs2::error& e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
