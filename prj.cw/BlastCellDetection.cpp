#include <BlastCellDetection.h>




namespace BlastCellDetection {

    PolygonVector::PolygonVector(const std::string& filepath) {
        cv::FileStorage fs(filepath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

        cv::FileNode root = fs.root();

        for (cv::FileNodeIterator it = root.begin(); it != root.end(); ++it) {
            cv::FileNode polygons = *it; // Извлекаем значение по ключу
            this->data.push_back(std::make_pair((*it).name(), std::vector<std::vector<cv::Point>>{}));

            for (cv::FileNodeIterator polygon = polygons.begin(); polygon != polygons.end(); ++polygon) {
                std::vector<cv::Point> tmp;
                cv::FileNode coords = *polygon; // Извлекаем значение по ключу

                for (int i = 0; i < coords.size(); i += 2) {
                    tmp.push_back(cv::Point{coords[i], coords[i + 1]});
                }

                this->data[this->data.size() - 1].second.push_back(tmp);
            }
        }

        fs.release();
    }


    Metrics::Metrics() {
        this->TP = this->FP = this->FN = 0.0;
    }


    double Metrics::getMean(const std::vector<double>& vec) {
        double tmp = 0;
        for (const double val : vec) {
            tmp += val;
        }

        return tmp / double(vec.size());
    }


    double Metrics::roundVal(double val) {
        return std::round(val * 100) / 100;
    }
    
    std::map<std::string, double> Metrics::getMetrics(const std::string& filePath) {
        double accuracy = this->TP / (this->TP + this->FP + this->FN), 
                precision = this->TP / (this->TP + this->FP),
                recall = this->TP / (this->TP + this->FN);

        std::map<std::string, double> results;

        results["accuracy-micro"] = this->roundVal(accuracy);
        results["accuracy-macro"] = this->roundVal(this->getMean(this->accuracyResults));

        results["precision-micro"] = this->roundVal(precision);
        results["precision-macro"] = this->roundVal(this->getMean(this->precisionResults));

        results["recall-micro"] = this->roundVal(recall);
        results["recall-macro"] = this->roundVal(this->getMean(this->recallResults));

        accuracy < 1e-3 ? results["f1-micro"] = 0.0 : results["f1-micro"] = this->roundVal(2 * precision * recall / (precision + recall));
        results["f1-macro"] = this->roundVal(this->getMean(this->f1Results));

        if (filePath.size()) {
            cv::FileStorage fs(filePath, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
            if (!fs.isOpened()) {
                throw std::runtime_error("Could not open the file to save JSON.");
            }

            fs << "metrics" << "{";
            for (const auto& kv : results) {
                fs << kv.first << kv.second;
            }
            fs << "}";

            fs.release();
        }

        return results;
    }


    void findCellColors(const cv::Mat& img, std::vector<cv::Vec3b>& cellColors) {
        // Переводим изображение в формат с плавающей точкой
        cv::Mat imgTmp;
        img.convertTo(imgTmp, CV_32F); 

        // Преобразуем вектора цветов для кластеризации
        cv::Mat data = imgTmp.reshape(1, img.cols * img.rows); 

        // K-Means кластеризация
        int k = 20; // Количество кластеров
        cv::Mat labels, centers;
        cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
            3, cv::KMEANS_PP_CENTERS, centers);

        // Преобразуем центры обратно в формат Vec3b
        centers.convertTo(centers, CV_8UC1);
        centers = centers.reshape(3, centers.rows);

        // Определим фиолетовый как цвет, ближе к которому будем искать кластеры
        std::vector<cv::Vec3b> targetColors = {{128, 0, 128}, {30, 30, 90}};
        for (const auto& targetColor : targetColors) {
            double minDistance = DBL_MAX;
            cv::Vec3b foundColor;
            for (int i = 0; i < centers.rows; i++) {
                cv::Vec3b color = centers.at<cv::Vec3b>(i, 0);
                double distance = cv::norm(color - targetColor);
                if (distance < minDistance) {
                    minDistance = distance;
                    foundColor = color;
                }
            }
            bool foundAlready = false;
            for (const auto& color : cellColors) {
                if (color == foundColor) {
                    foundAlready = true;
                    break;
                }
            }
            if (!foundAlready) {
                cellColors.push_back(foundColor);
            }
        }
    }


    std::pair<int, int> findChannelRange(const cv::Mat& channel, int targetValue) {
        // Построение гистограммы
        int histSize = 256; // from 0 to 255
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::Mat hist;
        cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        // Определение диапазона значений, близких к targetValue
        int highBound = targetValue, lowBound = targetValue;
        float threshold = hist.at<float>(targetValue) * 0.5; // Порог для определения значимых пикселей

        // Расширение диапазона с проверкой на границы
        while (highBound < 255 && hist.at<float>(highBound) > threshold) highBound++;
        while (lowBound > 0 && hist.at<float>(lowBound) > threshold) lowBound--;

        return std::make_pair(lowBound, highBound);
    }


    void detectBGR(const cv::Mat& originalImg, cv::Mat& detectionsImg) {
        detectionsImg = originalImg.clone();

        // Размытие для сглаживания и улучшения качества картинки
        cv::GaussianBlur(detectionsImg, detectionsImg, cv::Size(5, 5), 0);
        
        // Разделяем изображение на три канала
        std::vector<cv::Mat> channels;
        cv::split(detectionsImg, channels);

        // Поиск основных оттенков клетки
        std::vector<cv::Vec3b> cellColors;
        findCellColors(detectionsImg, cellColors);
        
        detectionsImg = cv::Mat::zeros(originalImg.size(), CV_8UC1);
        for (const auto& cellColor : cellColors) {
            // Находим эффективный диапазон для каждого канала
            cv::Vec3b lowBound, highBound;
            for (int c = 0; c < 3; c++) {
                std::pair<int, int> effectiveRange = findChannelRange(channels[c], cellColor[c]);
                lowBound[c] = effectiveRange.first;
                highBound[c] = effectiveRange.second;
            }

            // Маска для выделения пикселей в диапазоне
            cv::Mat mask;
            cv::inRange(originalImg, lowBound, highBound, mask);

            // Объединяем результаты в итоговое изображение
            detectionsImg |= mask;
        }

        // Морфологическое замыкание для заполнения маленьких дырочек
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(detectionsImg, detectionsImg, cv::MORPH_CLOSE, kernel);

        // Определение компонентов связности
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(detectionsImg, labels, stats, centroids);

        // Заполнение оставшихся дыр внутри компонент связности
        for (int i = 1; i < nLabels; i++) {      
            // Создаем маску для компонента
            cv::Mat componentMask = (labels == i);

            // Находим контуры компонента
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(componentMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Закрашиваем внутренние области компонента
            for (const auto& contour : contours) {
                cv::drawContours(detectionsImg, contours, -1, cv::Scalar(255), cv::FILLED);
            }
        }
    }


    void detectHSV(const cv::Mat& originalImg, cv::Mat& detectionsImg) {
        cv::Mat hsvImage;
        cv::GaussianBlur(originalImg, hsvImage, cv::Size(5, 5), 0);
        cv::cvtColor(hsvImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Scalar lowerHSV(90, 50, 50), upperHSV(180, 200, 200); // Нижний и верхний пределы (Hue, Saturation, Value)
        cv::inRange(hsvImage, lowerHSV, upperHSV, detectionsImg);

        // Закрытие дырочек
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(detectionsImg, detectionsImg, cv::MORPH_CLOSE, kernel);

        // Найти контуры
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(detectionsImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        // Собрать значения площади всех компонент связности
        std::vector<double> areas;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            areas.push_back(area);
        }

        // Отсортировать значения площади по возрастанию
        std::sort(areas.begin(), areas.end());

        // Найти порог для удаления компонент (10% от всех компонент)
        double threshold = areas[static_cast<int>(areas.size() * PERCENT_THRESHOLD / 100.0)];
        
        // Отрисовать хорошие компоненты связности
        detectionsImg = cv::Mat::zeros(detectionsImg.size(), CV_8UC1);
        for (int i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area >= threshold) {
                cv::drawContours(detectionsImg, contours, i, cv::Scalar(255), cv::FILLED, 8, hierarchy, 1);
            }
        }
    }


    void validate(const cv::Mat& detectionsImg, const std::vector<std::vector<cv::Point>>& groundTruth, Metrics& metrics) {
        double truePositive = 0.0, falseNegative = 0.0, falsePositive = 0.0;

        cv::Mat allGroundTruth = cv::Mat::zeros(detectionsImg.size(), CV_8UC1);

        // Поиск TP и FN        
        for (const auto& poly : groundTruth) {
            cv::Mat groundTruthPoly = cv::Mat::zeros(detectionsImg.size(), CV_8UC1);
            cv::drawContours(groundTruthPoly, std::vector<std::vector<cv::Point>>{poly}, -1, cv::Scalar(255), cv::FILLED);

            cv::Mat intersection;
            cv::bitwise_and(detectionsImg, groundTruthPoly, intersection);
            double intersectArea = cv::countNonZero(intersection);
            double groundTruthArea = cv::countNonZero(groundTruthPoly);

            intersectArea / groundTruthArea >= SQUARE_THRESHOLD ? truePositive++ : falseNegative++;

            cv::bitwise_or(allGroundTruth, groundTruthPoly, allGroundTruth); // Объединяем все ground truth полигоны
        }

        cv::Mat filteredDetections = cv::Mat::zeros(detectionsImg.size(), CV_8UC1);
        cv::bitwise_and(detectionsImg, ~allGroundTruth, filteredDetections); // Убираем TP и FN из детекции

        // Убираем мусорные ложные пиксели
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(~filteredDetections, filteredDetections, cv::MORPH_CLOSE, kernel);
        filteredDetections = ~filteredDetections;

        // Поиск FP
        std::vector<std::vector<cv::Point>> remainingContours;
        cv::findContours(filteredDetections, remainingContours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : remainingContours) {
            cv::Mat contourMask = cv::Mat::zeros(detectionsImg.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> singleContour{contour};
            cv::drawContours(contourMask, singleContour, -1, cv::Scalar(255), cv::FILLED);

            cv::Mat intersection;
            cv::bitwise_and(contourMask, allGroundTruth, intersection);

            if (cv::countNonZero(intersection) == 0) {
                falsePositive++;
            }
        }

        double accuracy = truePositive / (truePositive + falsePositive + falseNegative),
                precision = truePositive / (truePositive + falsePositive), 
                recall = truePositive / (truePositive + falseNegative);

        if (std::isnan(accuracy) || accuracy < 1e-3) accuracy = 0.0;
        if (std::isnan(precision) || precision < 1e-3) precision = 0.0;
        if (std::isnan(recall) || recall < 1e-3) recall = 0.0;

        metrics.TP += truePositive;
        metrics.FP += falsePositive;
        metrics.FN += falseNegative;
        metrics.accuracyResults.push_back(accuracy);
        metrics.precisionResults.push_back(precision);
        metrics.recallResults.push_back(recall);

        if (accuracy < 1e-3) metrics.f1Results.push_back(0.0);
        else metrics.f1Results.push_back(2 * precision * recall / (precision + recall));
    }


    cv::Mat drawContoursRed(const cv::Mat& detectionsImg) {
        cv::Mat result = detectionsImg.clone();
        
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(detectionsImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 2);
        }

        return result;
    }
}