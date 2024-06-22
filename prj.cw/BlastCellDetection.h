#ifndef BLASTCELLDETECTION_H
#define BLASTCELLDETECTION_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>


namespace BlastCellDetection {
    
    namespace {
        // Сколько процентов компонент связности убрать
        double PERCENT_THRESHOLD = 20.0; 

        // Минимальный порог пересечения groundTruth и prediction, чтобы засчитать truePositive
        double SQUARE_THRESHOLD = 0.8; 
    }

    struct PolygonVector {
        std::vector<std::pair<std::string, std::vector<std::vector<cv::Point>>>> data;
        PolygonVector(const std::string& filepath);
    };

    class Metrics {
        public:
            double TP, FP, FN;
            std::vector<double> accuracyResults, precisionResults, recallResults, f1Results;
            Metrics();
            std::map<std::string, double> getMetrics(const std::string& filePath = "");
        private:
            double getMean(const std::vector<double>& vec);
            double roundVal(double val);
    };

    void detectBGR(const cv::Mat& originalImg, cv::Mat& detectionsImg);

    void detectHSV(const cv::Mat& originalImg, cv::Mat& detectionsImg);

    cv::Mat drawContoursRed(const cv::Mat& img);

    void validate(
        const cv::Mat& detectionsImg, 
        const std::vector<std::vector<cv::Point>>& groundTruth, 
        Metrics& metrics
    );
}


#endif // BLASTCELLDETECTION_H