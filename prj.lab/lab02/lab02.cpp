#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>




// Generate img for tests
cv::Mat1b generateImg(const std::vector<int>& lvl) {
    cv::Mat1b img(256, 256, lvl[0]); // Background dark rectangle
    cv::rectangle(img, cv::Rect(23.5, 23.5, 209, 209), lvl[1], -1); // Second rectangle layer
    cv::circle(img, cv::Point(128, 128), 83, lvl[2], -1); // Circle layer

    return img;
}


// Draw histogram of the given img
cv::Mat drawHistogram(const cv::Mat& img) {
    cv::Mat1b histImg(256, 256, 230); // Background img

    // Calculations
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Histogram normalization
    cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

    // Displaying histogram
    for (int i = 0; i < 256; ++i) {
        cv::line(histImg, cv::Point(i, 255), cv::Point(i, 255 - cvRound(hist.at<float>(i))), cv::Scalar(0), 1);
    }

    return histImg;
}


// Change some pixels to add noise
cv::Mat addGaussianNoise(const cv::Mat& img, double stddev) {
    cv::Mat noisyImg = img.clone();
    cv::RNG rng; // Pseudorandom numbers module of the cv library

    // Noise generation
    cv::Mat noise(noisyImg.size(), CV_64FC1);
    rng.fill(noise, cv::RNG::NORMAL, 0, stddev);
    noisyImg += noise;

    return noisyImg;
}


int main() {
    std::vector<cv::Mat1b> imgs;
    std::vector<std::vector<int>> lvls = {
        {0, 127, 255},
        {20, 127, 235},
        {55, 127, 200},
        {90, 127, 165}
    };

    // Main action
    for (int i = 0; i < 4; i++) {
        cv::Mat1b img = generateImg(lvls[i]);
        cv::Mat1b lvlImg = img.clone();
        for (const auto& std : {3, 7, 15}) {
            cv::Mat1b noisyImg = addGaussianNoise(img, std);
            cv::vconcat(lvlImg, noisyImg, lvlImg);
            cv::vconcat(lvlImg, drawHistogram(noisyImg), lvlImg);
        }
        imgs.push_back(lvlImg);
    }

    // Gluing all imgs
    cv::Mat1b mainPic = imgs[0].clone();
    for (int i = 1; i < 4; i++) {
        cv::hconcat(mainPic, imgs[i], mainPic);
    }

    // Saving the img
    try {
        cv::imwrite("../export/lab02/result.png", mainPic);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    /*
    // Creating the report
    ReportCreator("lab02", 
    "1. write a function to generate a test img with three brightness levels"
    "(256 is the side of the img, 209 is the side of the inner square, 83 " 
    "is the radius of the circle)\n"
    "2. write a function to draw a brightness histogram on a square raster with "
    "a side of 256 in the form of columns with a width of 1px brightness 0 on a "
    "background of 230, normalize so that the maximum value has a height 230\n"
    "3. write a noise reduction function (additive normal unbiased noise with a "
    "given value of the standard deviation)\n"
    "4. generate test imgs for 4 sets of levels and glue them from left to right\n"
    "- [0,127,255]\n- [20,127,235]\n- [55,127,200]\n- [90,127,165]\n"
    "5. generate noisy imgs and histograms that are placed butt-to-butt below the test img\n"
    "6. make noise for three values of the standard deviation 3, 7, 15\n"
    "7. glue all imgs into one");
    */
   
    // Displaying the img
    cv::imshow("mainPic", mainPic);
    cv::waitKey(0);
}