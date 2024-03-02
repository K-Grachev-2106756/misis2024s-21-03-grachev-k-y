# Report for lab02
## Task:
1. write a function to generate a test image with three brightness levels(256 is the side of the image, 209 is the side of the inner square, 83 is the radius of the circle)
2. write a function to draw a brightness histogram on a square raster with a side of 256 in the form of columns with a width of 1px brightness 0 on a background of 230, normalize so that the maximum value has a height 230
3. write a noise reduction function (additive normal unbiased noise with a given value of the standard deviation)
4. generate test images for 4 sets of levels and glue them from left to right
- [0,127,255]
- [20,127,235]
- [55,127,200]
- [90,127,165]
5. generate noisy images and histograms that are placed butt-to-butt below the test image
6. make noise for three values of the standard deviation 3, 7, 15
7. glue all images into one
## Code:
```#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>



cv::Mat1b GenerateImage(const std::vector<int>& lvl) {
    cv::Mat1b img(256, 256, lvl[0]); // Background dark rectangle
    cv::rectangle(img, cv::Rect(23.5, 23.5, 209, 209), lvl[1], -1); // Second rectangle layer
    cv::circle(img, cv::Point(128, 128), 83, lvl[2], -1); // Circle layer

    return img;
}

cv::Mat DrawHistogram(const cv::Mat& image) {
    cv::Mat1b histImage(256, 256, 230); // Background img

    // Calculations
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Histogram normalization
    cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

    // Displaying histogram
    for (int i = 0; i < 256; ++i) {
        cv::line(histImage, cv::Point(i, 255), cv::Point(i, 255 - cvRound(hist.at<float>(i))), cv::Scalar(0), 1);
    }

    return histImage;
}

cv::Mat AddGaussianNoise(const cv::Mat& image, double stddev) {
    cv::Mat noisyImg = image.clone();
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
        cv::Mat1b img = GenerateImage(lvls[i]);
        cv::Mat1b lvlImg = img.clone();
        for (const auto& std : {3, 7, 15}) {
            cv::Mat1b noisyImg = AddGaussianNoise(img, std);
            cv::vconcat(lvlImg, noisyImg, lvlImg);
            cv::vconcat(lvlImg, DrawHistogram(noisyImg), lvlImg);
        }
        imgs.push_back(lvlImg);
    }

    // Gluing all images
    cv::Mat1b mainPic = imgs[0].clone();
    for (int i = 1; i < 4; i++) {
        cv::hconcat(mainPic, imgs[i], mainPic);
    }

    // Saving the image
    try {
        cv::imwrite("../export/lab02/result.png", mainPic);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    // Creating the report
    ReportCreator("lab02", 
    "1. write a function to generate a test image with three brightness levels"
    "(256 is the side of the image, 209 is the side of the inner square, 83 " 
    "is the radius of the circle)\n"
    "2. write a function to draw a brightness histogram on a square raster with "
    "a side of 256 in the form of columns with a width of 1px brightness 0 on a "
    "background of 230, normalize so that the maximum value has a height 230\n"
    "3. write a noise reduction function (additive normal unbiased noise with a "
    "given value of the standard deviation)\n"
    "4. generate test images for 4 sets of levels and glue them from left to right\n"
    "- [0,127,255]\n- [20,127,235]\n- [55,127,200]\n- [90,127,165]\n"
    "5. generate noisy images and histograms that are placed butt-to-butt below the test image\n"
    "6. make noise for three values of the standard deviation 3, 7, 15\n"
    "7. glue all images into one");

    // Displaying the image
    cv::imshow("mainPic", mainPic);
    cv::waitKey(0);
}
```
## Results:
!["result.png"](result.png)
