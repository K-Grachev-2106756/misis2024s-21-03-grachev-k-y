#include <BlastCellDetection.h>




int main() {
    cv::Mat img = cv::imread("../prj.cw/image.png"), detImg;

    BlastCellDetection::detect(img, detImg);
}