#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>
#include <filesystem>




bool isGrayscale(const cv::Mat& img) {
    cv::Mat b, g, r;
    std::vector<cv::Mat> channels = {b, g, r};
    cv::split(img, channels);

    // Подсчет различий между каналами
    cv::Mat diffRG, diffRB, diffGB;
    cv::absdiff(r, g, diffRG);
    cv::absdiff(r, b, diffRB);
    cv::absdiff(g, b, diffGB);

    int r_g = cv::countNonZero(diffRG);
    int r_b = cv::countNonZero(diffRB);
    int g_b = cv::countNonZero(diffGB);
    float diff_sum = static_cast<float>(r_g + r_b + g_b);

    std::cout << diff_sum;
    // Определение метки для изображения
    if (diff_sum / (img.rows * img.cols) > 0.005) {
        return false;
    }
    return true;
}

cv::Mat autoContrastImg(const cv::Mat& image, double clipHistPercent = 25.0) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hist;
    int histSize = 256; // Количество бинов в гистограмме
    float range[] = {0, 256}; // Диапазон значений пикселей
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    float accumulator[256] = {hist.at<float>(0)};
    for (int i = 1; i < 256; ++i) {
        accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
    }
    float maximum = accumulator[255];
    clipHistPercent *= (maximum / 200.0);

    int minimumGray = 0;
    while (accumulator[minimumGray] < clipHistPercent) {
        minimumGray++;
    }
    std::cout << minimumGray << " " << accumulator[minimumGray] << " " << maximum << " " << clipHistPercent << " " << std::endl;

    int maximumGray = 255;
    while (accumulator[maximumGray] >= maximum - clipHistPercent) {
        maximumGray--;
    }

    double alpha = 255.0 / (maximumGray - minimumGray);
    int beta = -minimumGray * alpha;
    
    cv::Mat res;
    cv::convertScaleAbs(image, res, alpha, beta);
    return res;
}

std::string getFileName(const std::string& filePath) {
    std::filesystem::path path(filePath);
    return path.filename().string();
}

//."C:/Users/user/Desktop/misis2024s-21-03-grachev-k-y/bin.dbg/lab03.exe" ..\prj.lab\lab03\1.jpg
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return 1;
    }
    cv::Mat img = cv::imread(argv[1]);
    bool isGray = isGrayscale(img);    
    std::string strGray = isGray ? "gray" : "color";

    cv::Mat contrasted = autoContrastImg(img);
    cv::hconcat(img, contrasted, img);

    cv::imwrite("../export/lab03/" + strGray + getFileName(argv[1]), img);
    cv::waitKey(0);
}

/*
    Бинаризация
    Заполнить дырочки
    Определить самые большие компоненты связности
    Задетектировать 
    Измерить качество масками.
*/