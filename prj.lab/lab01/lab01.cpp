#include <iostream>
#include <opencv2/opencv.hpp>


// Function for correction color level
int gammaCorrection(double color, double gamma) {
    return (int) (pow(color / 255, gamma) * 255);
}

int main(int argc, char** argv) {

    // Description of the parameters for calling the console application
    cv::CommandLineParser parser(argc, argv,
        "{@imageName    |      | output image name}"
        "{s             | 3    | gradient step width}"
        "{h             | 30   | gradient step height}"
        "{gamma         | 2.4  | gamma correction coef}"
    );
    
    // Parsing command line arguments
    cv::String imageName = parser.get<cv::String>("@imageName");
    int s = parser.get<int>("s");
    int h = parser.get<int>("h");
    double gamma = parser.get<double>("gamma");

    // Creating an image matrix
    cv::Mat1b img(2 * h, 256 * s, 1);

    // Filling matrix cells by colors
    for (int step = 0; step < 256; step++) { // For each color level (gradient step as a color)
        int correctedColor = gammaCorrection(step, gamma);
        std::cout<< correctedColor;
        for (int col = s * step; col < s * (step + 1); col++) { // For each col in rectangle
            for (int row = 0; row < h; row++) { // For each row in col of gradient rectangle
                img[row][col] = step;
            }
            for (int row = h; row < 2 * h; row++) { // For each row in col of gamma-corrected gradient rectangle
                img[row][col] = correctedColor;
            } 
        }
    }

    // Saving the image
    try {
        cv::imwrite("../export/lab01/lab01.png", img);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    // Displaying the image
    cv::imshow(imageName, img);
    cv::waitKey(0);
}