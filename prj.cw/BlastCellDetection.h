#ifndef BLASTCELLDETECTION_H
#define BLASTCELLDETECTION_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>




namespace BlastCellDetection {
    
    void findCellColor(const cv::Mat& img, cv::Vec3b& purpleColor);

    int findChannelRange(const cv::Mat& channel, int targetValue);

    void detect(cv::Mat& img);

}


#endif // BLASTCELLDETECTION_H