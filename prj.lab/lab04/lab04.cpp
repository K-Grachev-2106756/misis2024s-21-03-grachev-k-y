#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ReportCreator.h>




// Service class containing info about objects
class Circle {
    public:
        int x, y, radius, color;
        Circle(int x, int y, int radius, int color) {
            this->x = x;
            this->y = y;
            this->radius = radius;
            this->color = color;            
        }
};


// Make json file with info about circles
void makeJson(int width, int height, int backGroundColor, int blur, int stddev,
    const std::string& filePath, const std::vector<Circle>& circles) {
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs << "data" << "{"; 
    fs << "objects" << "[";
    for (const auto& circle : circles) {
        fs << "{"; 
        fs << "p" << "[" << circle.x << circle.y << circle.radius << "]";
        fs << "c" << circle.color;
        fs << "}";
    }
    fs << "]";
    fs << "background" << "{";
    fs << "size" << "[" << width << height << "]";
    fs << "color" << backGroundColor << "blur" << blur << "noise" << stddev << "amount" << (int)circles.size();
    fs << "}"; 
    fs << "}";

    fs.release();
}


// Get info from prepared json
void readJson(int& width, int& height, int& backGroundColor, int& blur, int& stddev,
    const std::string& filename, std::vector<Circle>& circles) {
    cv::FileStorage fs(filename, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    cv::FileNode data = fs["data"];

    cv::FileNode objects = data["objects"];
    for (cv::FileNodeIterator it = objects.begin(); it != objects.end(); ++it) {
        cv::FileNode obj = *it;
        cv::FileNode p = obj["p"];
        circles.push_back(Circle(p[0], p[1], p[2], obj["c"]));
    }

    cv::FileNode background = data["background"];
    width = background["size"][0], height = background["size"][1], backGroundColor = background["color"],
        blur = background["blur"], stddev = background["noise"];

    fs.release();
}


// Gen image with noise and blur
cv::Mat genImg(int width, int height, int backGroundColor, int blur, int stddev, const std::vector<Circle>& circles) {
    cv::Mat result = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(backGroundColor));
    
    int maxRadius = std::max_element(
        circles.begin(), circles.end(), [](const Circle &a, const Circle &b) {return a.radius < b.radius;}
    )->radius; // Searching for the maximum radius of the circle
    int blurScopeSize = maxRadius * 3; // Size of the blur area
    if (blur % 2 == 0) blur++; // Setting the blur parameter (odd value)
    for (const auto& circle : circles) {
        cv::circle(result, cv::Point(circle.x, circle.y), circle.radius, cv::Scalar(circle.color), cv::FILLED);

        // Definition of the blur area
        int xTopAngle = std::max(circle.x - blurScopeSize / 2, 0),
            yTopAngle = std::max(circle.y - blurScopeSize / 2, 0),
            blurWidth = std::min(blurScopeSize, width - xTopAngle),
            blurHeight = std::min(blurScopeSize, height - yTopAngle);
        cv::Rect blurScope(xTopAngle, yTopAngle, blurWidth, blurHeight);

        // Applying Gaussian blur to the selected area
        cv::Mat blurScopeLink = result(blurScope);
        cv::GaussianBlur(blurScopeLink, blurScopeLink, cv::Size(blur, blur), 0);
    }

    // Creating and adding noise to an image
    cv::Mat noise = cv::Mat(height, width, CV_8UC1);
    cv::randn(noise, 0, stddev);
    result += noise;

    return result;
}


int main() {
    // Main parameters
    int circlesInRow, minRadius, maxRadius, minColor, maxColor, stddev, blur, backGroundColor, width, height;
    std::vector<Circle> circles;

    if (false) {
        circlesInRow = 2, minRadius = 10, maxRadius = 25, minColor = 127, maxColor = 255, 
        stddev = 5, blur = 33, backGroundColor = 90, width = 350, height = 350;

        // Genering info about objects
        int circleColor = minColor, 
            coordinateStep = 3 * maxRadius, 
            colorStep = (maxColor - minColor) / circlesInRow, 
            imgSize = circlesInRow * coordinateStep,
            current_x = coordinateStep / 2, 
            current_y = coordinateStep / 2;
        while (current_y + maxRadius <= imgSize) {
            for (int i = 0; i < circlesInRow; i++) {
                int radius = minRadius + i * (maxRadius - minRadius) / (circlesInRow - 1);
                circles.push_back(Circle(current_x, current_y, radius, circleColor));
                current_x += coordinateStep;
                if (current_x >= imgSize) {
                    current_x = coordinateStep / 2;
                    current_y += coordinateStep;
                }
            }
            circleColor += colorStep;
        }

        // Saving inf
        makeJson(width, height, backGroundColor, blur, stddev, "../export/lab04/ground_truth.json", circles);
    } else {
        // Loading inf
        readJson(width, height, backGroundColor, blur, stddev, "../export/lab04/ground_truth.json", circles);

        cv::imshow("", genImg(width, height, backGroundColor, blur, stddev, circles));
        cv::waitKey(0);
    }

    

}