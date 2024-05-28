#include <BlastCellDetection.h>




namespace BlastCellDetection {

    void findCellColor(const cv::Mat& img, cv::Vec3b& purpleColor) {
        // Переводим изображение в формат с плавающей точкой
        cv::Mat imgTmp;
        img.convertTo(imgTmp, CV_32F); 

        // Преобразуем вектора цветов для кластеризации
        cv::Mat data = imgTmp.reshape(1, img.cols * img.rows); 

        // K-Means кластеризация
        int k = 4; // Количество кластеров
        cv::Mat labels, centers;
        cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
            3, cv::KMEANS_PP_CENTERS, centers);

        // Преобразуем центры обратно в формат Vec3b
        centers = centers.reshape(3, centers.rows);

        // Определим фиолетовый как цвет, ближе к которому будем искать кластеры
        cv::Vec3b targetPurple(128, 0, 128); // Примерный фиолетовый цвет в BGR
        double minDistance = DBL_MAX;
        for (int i = 0; i < centers.rows; i++) {
            cv::Vec3b color = centers.at<cv::Vec3f>(i, 0);
            double distance = cv::norm(color - targetPurple);
            if (distance < minDistance) {
                minDistance = distance;
                purpleColor = color;
            }
        }
    }


    int findChannelRange(const cv::Mat& channel, int targetValue) {
        // Построение гистограммы
        int histSize = 256; // from 0 to 255
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::Mat hist;
        cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        // Определение диапазона значений, близких к targetValue
        int highBound = targetValue;
        float threshold = hist.at<float>(targetValue) * 0.5; // Порог для определения значимых пикселей

        // Расширение диапазона
        while (hist.at<float>(highBound) > threshold && highBound < 255) highBound++;

        /*
        // Histogram normalization
        cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

        // Displaying histogram
        cv::Mat1b histImg(256, 256, 230); // Background img
        for (int i = 0; i < 256; ++i) {
            cv::line(histImg, cv::Point(i, 255), cv::Point(i, 255 - cvRound(hist.at<float>(i))), cv::Scalar(0), 1);
        }
        std::cout << std::endl << std::endl << std::endl << std::endl << highBound << " " << targetValue << " " << highBound << std::endl << std::endl << std::endl << std::endl;

        cv::imshow("Original Image", histImg);
        cv::waitKey(0);
        */

        return highBound;
    }


    void detect(cv::Mat& img) {
        cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
        cv::Vec3b purpleColor;
        findCellColor(img, purpleColor);

        // Разделяем изображение на три канала
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        // Находим эффективный диапазон для каждого канала
        cv::Vec3b lowBoundPurple = {0, 0, 0}, highBoundPurple;
        for (int i = 0; i < 3; i++) {
            highBoundPurple[i] = findChannelRange(channels[i], purpleColor[i]);
        }

        // Пороговое значение для бинаризации
        cv::Mat binary;
        cv::inRange(img, lowBoundPurple, highBoundPurple, binary);

        cv::imshow("Original Image", binary);
        cv::waitKey(0);

        return;
        // Проходим по всем пикселям изображения и заменяем целевой цвет на новый
        cv::Vec3b newColor(0, 0, 255); 
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                if (img.at<cv::Vec3b>(y, x) == purpleColor) {
                    img.at<cv::Vec3b>(y, x) = newColor;
                }
            }
        }
    }
}