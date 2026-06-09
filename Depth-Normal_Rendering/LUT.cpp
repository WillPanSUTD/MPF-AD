#include "LUT.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

LUT::LUT() : depthHistogram(), colorLUT(), depthMin(0), depthMax(0) {}
LUT::~LUT() {}

// Build the depth histogram with Freedman-Diaconis bin width
// (h = 2 * IQR / N^{1/3}). Pixels with depth < -1e30 are treated as invalid
// and excluded from both the bin-count and the histogram itself.
bool LUT::buildDepthHistogramIQR(const cv::Mat& depthMap, int bins)
{
    if (depthMap.empty()) {
        std::cerr << "Error: input depth map is empty." << std::endl;
        return false;
    }
    if (depthMap.channels() != 1 || depthMap.type() != CV_32FC1) {
        std::cerr << "Error: only single-channel 32-bit float depth maps (CV_32FC1) are supported." << std::endl;
        return false;
    }

    // Mask out invalid pixels (sentinel ~ -3.4e38).
    cv::Mat mask = (depthMap > -1e30);
    int validPixelCount = cv::countNonZero(mask);
    if (validPixelCount == 0) {
        std::cerr << "Warning: no valid pixels in the depth map." << std::endl;
        depthHistogram.release();
        return false;
    }

    // Convert valid depths to double for accurate quartile computation.
    cv::Mat valid64f;
    depthMap.convertTo(valid64f, CV_64F);
    valid64f.setTo(0, ~mask);

    // Collect valid values and sort them so we can pick out Q1, Q3.
    std::vector<double> validValues;
    validValues.reserve(validPixelCount);
    for (int y = 0; y < valid64f.rows; y++) {
        const double* row = valid64f.ptr<double>(y);
        const uchar*  m   = mask.ptr<uchar>(y);
        for (int x = 0; x < valid64f.cols; x++) {
            if (m[x]) validValues.push_back(row[x]);
        }
    }
    std::sort(validValues.begin(), validValues.end());

    // Compute Q1, Q3, IQR by linear interpolation between order statistics.
    double q1, q3, iqr;
    size_t n = validValues.size();
    if (n <= 1) {
        q1 = q3 = validValues.empty() ? 0 : validValues[0];
        iqr = 0;
    }
    else {
        auto getQ = [&](double p) {
            double pos = p * (n - 1);
            int idx = (int)pos;
            double f = pos - idx;
            return validValues[idx] + f * (validValues[idx + 1] - validValues[idx]);
            };
        q1 = getQ(0.25);
        q3 = getQ(0.75);
        iqr = q3 - q1;
    }

    cv::minMaxLoc(valid64f, &depthMin, &depthMax, nullptr, nullptr, mask);
    if (depthMin >= depthMax) {
        // Degenerate: the depth map is essentially flat. Return a 1-bin histogram.
        depthHistogram = cv::Mat::zeros(1, 1, CV_64F);
        depthHistogram.at<double>(0) = n;
        return true;
    }

    // Auto bin count via Freedman-Diaconis when bins == -1.
    if (bins == -1) {
        if (iqr < 1e-6) bins = 1;
        else {
            double h = 2 * iqr / pow(n, 1.0 / 3.0);
            bins = (int)ceil((depthMax - depthMin) / h);
            bins = std::max(bins, 1);
        }
    }
    else if (bins <= 0) {
        std::cerr << "Error: bin count must be positive." << std::endl;
        return false;
    }

    // Accumulate the histogram.
    depthHistogram = cv::Mat::zeros(1, bins, CV_64F);
    double binW = (depthMax - depthMin) / bins;

    const float* d = depthMap.ptr<float>();
    const uchar* m = mask.ptr<uchar>();
    for (size_t i = 0; i < depthMap.total(); i++) {
        if (!m[i]) continue;
        double v = (double)d[i];
        int b = (int)((v - depthMin) / binW);
        b = std::min(b, bins - 1);
        depthHistogram.at<double>(b)++;
    }
    return true;
}

// Build a 256-entry blue -> green -> red gradient.
// First half: blue fading to green; second half: green fading to red.
bool LUT::createColorLUT() {
    if (depthHistogram.empty() || depthMin == depthMax) {
        std::cerr << "Error: build the depth histogram first." << std::endl;
        return false;
    }

    colorLUT.create(1, 256, CV_8UC3);
    cv::Vec3b* p = colorLUT.ptr<cv::Vec3b>(0);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uchar r = 0, g = 0, b = 0;
        if (t < 0.5f) {
            // First half: blue (255,0,0 in BGR order) -> green (0,255,0).
            float f = t * 2;
            b = (uchar)((1 - f) * 255);
            g = (uchar)(f * 255);
        }
        else {
            // Second half: green -> red (0,0,255).
            float f = (t - 0.5f) * 2;
            g = (uchar)((1 - f) * 255);
            r = (uchar)(f * 255);
        }
        p[i] = cv::Vec3b(b, g, r);
    }
    return true;
}

// Map every valid depth pixel to a BGR color via the LUT.
// Implementation note: we min-max normalize the valid depth range to [0, 255]
// and then look up the corresponding LUT entry.
bool LUT::applyColorMap(const cv::Mat& depthMap, cv::Mat& colorMap) const
{
    if (depthMap.empty() || depthMap.channels() != 1 || depthMap.type() != CV_32FC1) {
        std::cerr << "Error: a 32-bit float depth map (CV_32FC1) is required." << std::endl;
        return false;
    }
    if (colorLUT.empty() || colorLUT.type() != CV_8UC3 || colorLUT.cols != 256) {
        std::cerr << "Error: the color LUT is invalid." << std::endl;
        return false;
    }

    cv::Mat mask = (depthMap > -1e30);
    cv::Mat valid;
    depthMap.copyTo(valid, mask);

    // Normalize valid depth to [0, 255].
    double minv, maxv;
    cv::minMaxLoc(valid, &minv, &maxv, nullptr, nullptr, mask);

    cv::Mat norm8u;
    if (maxv - minv < 1e-6)
        norm8u = cv::Mat::zeros(depthMap.size(), CV_8U);
    else
        cv::normalize(valid, norm8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply the LUT in BGR space.
    cv::Mat c3;
    cv::cvtColor(norm8u, c3, cv::COLOR_GRAY2BGR);
    cv::LUT(c3, colorLUT, colorMap);

    // Force invalid pixels to black.
    colorMap.setTo(0, ~mask);
    return true;
}

const cv::Mat& LUT::getDepthHistogram() const { return depthHistogram; }
const cv::Mat& LUT::getColorLUT()       const { return colorLUT; }

// Render the histogram as a bar chart on a white canvas.
void LUT::visualizeHistogram(cv::Mat& histImage, int w, int h) const {
    if (depthHistogram.empty()) { histImage.release(); return; }
    histImage = cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    double maxH; cv::minMaxLoc(depthHistogram, nullptr, &maxH);
    int bins = depthHistogram.cols;
    int m = 20, bh = h - 2 * m;
    for (int i = 0; i < bins; i++) {
        double v = depthHistogram.at<double>(i);
        int hi = cvRound(v / maxH * bh);
        int x = m + i * (w - 2 * m) / bins;
        cv::rectangle(histImage, { x, h - m - hi, (w - 2 * m) / bins - 1, hi }, { 200,0,0 }, -1);
    }
}

// Render the color LUT as a vertical color bar of the given width.
void LUT::visualizeColorLUT(cv::Mat& lutImage, int bw) const {
    if (colorLUT.empty() || colorLUT.cols != 256) { lutImage.release(); return; }
    lutImage.create(256, bw, CV_8UC3);
    const cv::Vec3b* p = colorLUT.ptr<cv::Vec3b>();
    for (int i = 0; i < 256; i++) lutImage.row(i) = p[i];
}
