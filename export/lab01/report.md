# Report for lab01
## Task:
1. write a console application to generate a single-channel 8bpp image with a "gradient fill"(from 0 to 255, from rectangles s-width, h-height) and gamma-corrected fill
2. strips are arranged on top of each other
3. gamma correction is executed as a function
4. the name of the output file is specified as an optional parameter (without a key) if this parameter is not set, then just show the result on the screen and close the application by pressing any key
5. get the s, h, gamma parameters from the command line parameters, if the keys are not specified, then use the defaults (s=3, h=30, gamma=2.4)
## Code:
```#include <iostream>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>




// Function for correction color level
int GammaCorrection(double color, double gamma) {
    return (int) (pow(color / 255, gamma) * 255);
}

int main(int argc, char** argv) {

    // Description of the parameters for calling the console application
    cv::CommandLineParser parser(argc, argv,
        "{imageName   |      | output image name}"
        "{s           | 3    | gradient step width}"
        "{h           | 30   | gradient step height}"
        "{gamma       | 2.4  | gamma correction coef}"
    );
    
    // Parsing command line arguments
    cv::String imageName = parser.get<cv::String>("imageName");
    int s = parser.get<int>("s");
    int h = parser.get<int>("h");
    double gamma = parser.get<double>("gamma");

    std::cout<< imageName << std::endl;
    std::cout<< s << std::endl;
    std::cout<< h << std::endl;
    std::cout<< gamma << std::endl;

    // Creating an image matrix
    cv::Mat1b img(2 * h, 256 * s, 1);

    // Filling matrix cells by colors
    for (int step = 0; step < 256; step++) { // For each color level (gradient step as a color)
        int correctedColor = GammaCorrection(step, gamma);
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
        std::string name = imageName == "" ? "default" : imageName;
        cv::imwrite("../export/lab01/" + name + ".png", img);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    // Creating the report
    ReportCreator("lab01", 
    "1. write a console application to generate a single-channel 8bpp image with a \"gradient fill\"" 
    "(from 0 to 255, from rectangles s-width, h-height) and gamma-corrected fill\n"
    "2. strips are arranged on top of each other\n"
    "3. gamma correction is executed as a function\n"
    "4. the name of the output file is specified as an optional parameter (without a key) if this parameter is not set, "
    "then just show the result on the screen and close the application by pressing any key\n"
    "5. get the s, h, gamma parameters from the command line parameters, if the keys are not specified, "
    "then use the defaults (s=3, h=30, gamma=2.4)");

    // Displaying the image
    cv::imshow(imageName, img);
    cv::waitKey(0);
}
```
## Results:
![default.png](default.png)
![gamma.png](gamma.png)
![width.png](width.png)
