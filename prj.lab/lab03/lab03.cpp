#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>



//."C:/Users/user/Desktop/misis2024s-21-03-grachev-k-y/bin.dbg/lab03.exe" ..\prj.lab\lab03\1.jpg
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);

    std::cout << std::endl << img.channels() << std::endl << std::endl;
    
    cv::imshow("", img);
    cv::waitKey(0);
}