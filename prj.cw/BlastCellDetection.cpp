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
        std::cout << std::endl << std::endl << highBound << " " << targetValue << " " << highBound << std::endl << std::endl;

        cv::imshow("Original Image", histImg);
        cv::waitKey(0);
        */

        return highBound;
    }


    void detect(const cv::Mat& originalImg, cv::Mat& detectionsImg) {
        detectionsImg = originalImg.clone();

        // Размытие для сглаживания и улучшения качества картинки
        cv::GaussianBlur(detectionsImg, detectionsImg, cv::Size(5, 5), 0);
        
        // Поиск оттенка, который ближе всего к фиолетовому
        cv::Vec3b purpleColor;
        findCellColor(detectionsImg, purpleColor);

        // Разделяем изображение на три канала
        std::vector<cv::Mat> channels;
        cv::split(detectionsImg, channels);
        
        // Находим эффективный диапазон для каждого канала
        cv::Vec3b lowBoundPurple = {0, 0, 0}, highBoundPurple;
        for (int i = 0; i < 3; i++) {
            highBoundPurple[i] = findChannelRange(channels[i], purpleColor[i]);
        }

        // Бинаризация
        cv::inRange(detectionsImg, lowBoundPurple, highBoundPurple, detectionsImg);

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
        cv::imshow("Original Image", detectionsImg);
        cv::waitKey(0);
        /*
        // Нахождение компонентов связности
        // Объединение изображений вертикально
        cv::Mat combined;
        cv::vconcat(std::vector<cv::Mat>{binary, binary}, combined);

        cv::imshow("Original Image", combined);
        cv::waitKey(0);

        return;
        
        cv::Mat labels;
        int numComponents = connectedComponents(binary, labels, 8, CV_32S);

        // Создание цветного изображения для визуализации
        cv::Mat output = cv::Mat::zeros(binary.size(), CV_8UC3);

        // Генерация случайных цветов для каждой компоненты
        cv::RNG rng(12345);
        std::vector<cv::Vec3b> colors(numComponents);
        colors[0] = cv::Vec3b(0, 0, 0); // Цвет для фона
        for (int i = 1; i < numComponents; ++i) {
            colors[i] = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }

        // Окрашивание компонент связности
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                int label = labels.at<int>(y, x);
                output.at<cv::Vec3b>(y, x) = colors[label];
            }
        }*/
    }
}