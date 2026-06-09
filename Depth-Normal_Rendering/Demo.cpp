// Reference driver for the depth-normal guided illumination renderer.
//
// Usage:
//     depth_normal_render <input.tif> <output_dir>
//
// Produces three files in <output_dir>:
//     depth_lut.bmp   - Stage 1a pseudocolor image from the statistical-prior LUT
//     normals.bmp     - per-pixel surface normals visualized in BGR
//     phong.bmp       - final rendered image (Stage 1a x Phong shading)
//
// The default Phong parameters reproduce the values reported in the paper:
//     L = V = (500, 150, 1500), I_l = 1.3,
//     w_a = 0.1, w_d = 0.5, w_s = 0.3, shininess = 32.
//
// Edit the constants below or extend the CLI if you need to sweep parameters.

#include "Phong.h"
#include "LUT.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static int run(const std::string& depthPath, const std::string& outDir)
{
    cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    if (depthMap.empty() || depthMap.type() != CV_32FC1) {
        std::cerr << "Error: a single-channel 32-bit float depth map (CV_32FC1) is required."
                  << "\n  Got: " << depthPath << std::endl;
        return -1;
    }

    // ---- Stage 1a: statistical-prior LUT pseudocolor ----
    LUT depthLUT;
    if (!depthLUT.buildDepthHistogramIQR(depthMap) || !depthLUT.createColorLUT()) {
        std::cerr << "Error: failed to build the LUT from the depth histogram." << std::endl;
        return -1;
    }

    cv::Mat depthColorMap;
    if (!depthLUT.applyColorMap(depthMap, depthColorMap)) {
        std::cerr << "Error: failed to apply the color LUT." << std::endl;
        return -1;
    }
    cv::imwrite(outDir + "/depth_lut.bmp", depthColorMap);

    // ---- Stage 1b: surface normals from depth ----
    Phong phong;
    cv::Mat mask = (depthMap > -1e30);
    if (!phong.computeNormals(depthMap, /*use8Neighbors=*/false)) {
        std::cerr << "Error: surface normal estimation failed." << std::endl;
        return -1;
    }

    cv::Mat normalsVis;
    phong.visualizeNormals(normalsVis);
    normalsVis.setTo(cv::Scalar(0, 0, 0), ~mask);
    cv::imwrite(outDir + "/normals.bmp", normalsVis);

    // ---- Stage 1c: Phong shading modulating the LUT pseudocolor ----
    cv::Vec3f light(500.0f, 150.0f, 1500.0f);  // L
    cv::Vec3f view (500.0f, 150.0f, 1500.0f);  // V
    float ambient        = 0.1f;   // w_a
    float lightIntensity = 1.3f;   // I_l
    float kd             = 0.5f;   // w_d (diffuse)
    float ks             = 0.3f;   // w_s (specular)
    float shine          = 32.0f;  // n   (specular exponent)

    cv::Mat shadedPhong;
    if (!phong.renderPhong(depthColorMap, shadedPhong, light, view,
                           lightIntensity, ambient, kd, ks, shine, mask)) {
        std::cerr << "Error: Phong rendering failed." << std::endl;
        return -1;
    }
    cv::imwrite(outDir + "/phong.bmp", shadedPhong);

    std::cout << "Wrote " << outDir << "/{depth_lut,normals,phong}.bmp" << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "depth_normal_render")
                  << " <input.tif> <output_dir>" << std::endl;
        return -1;
    }
    std::string depthPath = argv[1];
    std::string outDir    = argv[2];

    // Make sure the output directory exists.
    fs::create_directories(outDir);

    return run(depthPath, outDir);
}
