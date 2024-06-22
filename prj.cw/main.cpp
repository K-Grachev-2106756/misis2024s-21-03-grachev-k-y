#include <BlastCellDetection.h>
#include <filesystem>
#include <fstream>

using namespace BlastCellDetection;
namespace fs = std::filesystem;




int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv,
        "{imgPath         |      | path to the image}"
        "{groundTruthPath |      | path to the ground truth file}"
        "{mode            |      | detection mode (BGR or HSV)}"
    );

    std::string imgPath = parser.get<std::string>("imgPath");
    std::string groundTruthPath = parser.get<std::string>("groundTruthPath");
    std::string mode = parser.get<std::string>("mode");

    if (mode != "BGR" && mode != "HSV" && mode != "") {
        std::cerr << "Invalid mode. Running both BGR and HSV modes." << std::endl;
        mode = "";
    }

    if (!groundTruthPath.empty()) {
        std::ifstream gtFile(groundTruthPath);
        if (!gtFile.is_open()) {
            std::cerr << "Error: Could not open the ground truth file." << std::endl;
            return -1;
        }
    }

    if (imgPath.empty()) {
        const std::string path = "../blast_cell_dataset/";

        if (!fs::exists(path)) {
            std::cerr << "Dataset not found. Please download the dataset from GitHub." << std::endl;
            return -1;
        } else {
            PolygonVector polygons = PolygonVector(path + "ground_truth.json");
            Metrics metricsBGR, metricsHSV;

            if (mode == "HSV" ||  mode == "") {
                std::cout << "MODE HSV\n[";
                for (auto& picture_data : polygons.data) {
                    std::cout << "=";
                    cv::Mat img = cv::imread("../blast_cell_dataset/images/" + picture_data.first + ".jpg"), detImg;
                    detectHSV(img, detImg);
                    validate(detImg, picture_data.second, metricsBGR);
                }
                std::cout << "]";
                metricsBGR.getMetrics(path + "HSV.json");
            }

            if (mode == "BGR" ||  mode == "") {
                std::cout << "MODE BGR\n[";
                for (auto& picture_data : polygons.data) {
                    std::cout << "=";
                    cv::Mat img = cv::imread("../blast_cell_dataset/images/" + picture_data.first + ".jpg"), detImg;
                    detectBGR(img, detImg);
                    validate(detImg, picture_data.second, metricsBGR);
                }
                std::cout << "]";
                metricsBGR.getMetrics(path + "BGR.json");
            }
            return 0;
        }
    }

    cv::Mat originalImg = cv::imread(imgPath), detImg;
    if (originalImg.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    std::filesystem::path path(imgPath);
    std::filesystem::path directory = path.parent_path();

    if (mode == "HSV" || mode == "") {
        detectHSV(originalImg, detImg);
        std::filesystem::path newFilePath = directory / "HSV.png";
        cv::imwrite(newFilePath.string(), drawContoursRed(detImg));
        
        if (!groundTruthPath.empty()) {
            Metrics metricsHSV;
            PolygonVector polygons = PolygonVector(groundTruthPath);
            validate(detImg, polygons.data[0].second, metricsHSV);

            newFilePath = directory / "HSV.json";
            metricsHSV.getMetrics(newFilePath.string());
        }
    }

    if (mode == "BGR" || mode == "") {
        detectBGR(originalImg, detImg);
        std::string newFileName = "BGR.png";
        std::filesystem::path newFilePath = directory / newFileName;
        cv::imwrite(newFilePath.string(), drawContoursRed(detImg));

        if (!groundTruthPath.empty()) {
            Metrics metricsBGR;
            PolygonVector polygons = PolygonVector(groundTruthPath);
            validate(detImg, polygons.data[0].second, metricsBGR);

            newFilePath = directory / "BGR.json";
            metricsBGR.getMetrics(newFilePath.string());
        }
    }
}