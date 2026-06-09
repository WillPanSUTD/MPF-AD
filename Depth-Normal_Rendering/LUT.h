// Header guard.
#ifndef LUT_H
#define LUT_H

#include <opencv2/opencv.hpp>

/**
 * @brief Depth-to-pseudocolor LUT with statistical-prior bin width.
 *
 * Converts a single-channel 32-bit float depth map (CV_32FC1, sentinel
 * value < -1e30 marks invalid pixels) into an 8-bit BGR pseudocolor image.
 * The histogram bin width is chosen by the Freedman-Diaconis rule on the
 * inter-quartile range of valid depth values, so subtle defect-induced
 * relative-depth differences are preserved in the color encoding.
 */
class LUT
{
private:
    cv::Mat depthHistogram;     // Per-bin pixel count over valid depths.
    cv::Mat colorLUT;           // 256-entry BGR lookup table (CV_8UC3).
    double  depthMin, depthMax; // Min / max of valid depth values.

public:
    LUT();
    ~LUT();

    // Build the depth histogram. With bins = -1 the Freedman-Diaconis rule
    // chooses the number of bins automatically from the IQR of valid depths.
    bool buildDepthHistogramIQR(const cv::Mat& depthMap, int bins = -1);

    // Build the 256-entry blue->green->red gradient lookup table.
    bool createColorLUT();

    // Apply the LUT to a depth map; output is an 8-bit BGR pseudocolor image.
    bool applyColorMap(const cv::Mat& depthMap, cv::Mat& colorMap) const;

    // Accessors.
    const cv::Mat& getDepthHistogram() const;
    const cv::Mat& getColorLUT() const;

    // Render the depth histogram as an image (for debugging / paper figures).
    void visualizeHistogram(cv::Mat& histImage, int canvasWidth, int canvasHeight) const;

    // Render the color LUT as a vertical color bar (for debugging).
    void visualizeColorLUT(cv::Mat& lutImage, int barWidth = 150) const;
};

#endif
