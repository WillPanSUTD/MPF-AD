#include "Phong.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

// L2-normalize in place; vectors below 1e-6 are replaced with (0, 0, 1).
void Phong::normalizeNormal(cv::Vec3f& normal)
{
    float norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (norm < 1e-6) {
        normal = cv::Vec3f(0, 0, 1);
    }
    else {
        normal /= norm;
    }
}

// PCA on a 3D point set: the plane normal is the eigenvector of the
// covariance matrix corresponding to the smallest eigenvalue.
bool Phong::computeEigenVector(const std::vector<cv::Vec3f>& points, cv::Vec3f& normal)
{
    if (points.size() < 3) {
        return false;
    }

    // Centroid.
    cv::Vec3f mean(0, 0, 0);
    for (const auto& p : points) mean += p;
    mean /= static_cast<float>(points.size());

    // Covariance matrix.
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto& p : points) {
        Eigen::Vector3f vec(p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]);
        cov += vec * vec.transpose();
    }
    cov /= static_cast<float>(points.size() - 1);

    // Eigendecomposition.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    // Eigenvector for the smallest eigenvalue (column 0) = plane normal.
    normal = cv::Vec3f(
        solver.eigenvectors().col(0)[0],
        solver.eigenvectors().col(0)[1],
        solver.eigenvectors().col(0)[2]
    );
    return true;
}

// Auto-scale Z so depth differences (mm) and pixel offsets (px) have
// comparable magnitudes -> normals stay numerically well-conditioned.
float Phong::computeDepthScale(const cv::Mat& depth32F)
{
    if (depth32F.empty() || depth32F.type() != CV_32FC1)
        return 1.0f;

    cv::Mat mask = (depth32F > -1e30);
    double minv, maxv;
    cv::minMaxLoc(depth32F, &minv, &maxv, nullptr, nullptr, mask);

    float diff = (float)(maxv - minv);
    if (diff < 1e-6)
        return 1.0f;

    return 512.0f / diff;
}

// Boundary fallback: use the surface normal from the nearby pixel with the
// largest depth difference (search radius 1).
cv::Vec3f Phong::getNearestValidNormalWithDepthDiff(const cv::Mat& depthMap, int x, int y,
                                                    float centerDepth, int rows, int cols)
{
    const int searchRadius = 1;
    cv::Vec3f bestNormal(0, 0, 1);
    float maxDiff = -1;

    for (int r = 1; r <= searchRadius; ++r) {
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                // Only walk the boundary of the current search radius.
                if (std::abs(dx) != r && std::abs(dy) != r) continue;

                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;

                float nDepth = depthMap.at<float>(ny, nx);
                if (nDepth < -1e30) continue;             // skip invalid depths

                cv::Vec3f nNormal = normals.at<cv::Vec3f>(ny, nx);
                if (nNormal == cv::Vec3f(0, 0, 0)) continue;  // skip invalid normals

                float diff = std::abs(nDepth - centerDepth);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    bestNormal = nNormal;
                }
            }
        }
        if (maxDiff > 0) break;
    }
    return bestNormal;
}

// Estimate per-pixel surface normals.
//   1. Build a list of valid (dx, dy, scale * dz) neighbors.
//   2. Use PCA when >= 3 valid neighbors, cross product when exactly 2.
//   3. Otherwise fall back to the boundary helper.
//   4. Final normals are L2-normalized and forced to face the camera (nz >= 0).
bool Phong::computeNormals(const cv::Mat& depthMap, bool use8Neighbors)
{
    if (depthMap.empty() || depthMap.channels() != 1 || depthMap.type() != CV_32FC1) {
        std::cerr << "Error: depth map must be CV_32FC1 (single-channel 32-bit float)." << std::endl;
        return false;
    }

    float scale = computeDepthScale(depthMap);
    int rows = depthMap.rows;
    int cols = depthMap.cols;
    normals = cv::Mat::zeros(rows, cols, CV_32FC3);

    std::vector<cv::Point> neighbors;
    if (use8Neighbors) {
        neighbors = {
            {-1, -1}, { 0, -1}, { 1, -1},
            {-1,  0},          { 1,  0},
            {-1,  1}, { 0,  1}, { 1,  1}
        };
    }
    else {
        neighbors = { { 0, -1}, {-1, 0}, {1, 0}, {0, 1} };
    }

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float centerDepth = depthMap.at<float>(y, x);

            if (centerDepth < -1e30) {
                normals.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                continue;
            }

            std::vector<cv::Vec3f> validPoints;
            for (const auto& off : neighbors) {
                int nx = x + off.x;
                int ny = y + off.y;
                if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;

                float nDepth = depthMap.at<float>(ny, nx);
                if (nDepth < -1e30) continue;

                float dx = (float)off.x;
                float dy = (float)off.y;
                float dz = (nDepth - centerDepth) * scale;
                validPoints.emplace_back(dx, dy, dz);
            }

            cv::Vec3f normal;
            bool success = false;

            if (validPoints.size() >= 3) {
                success = computeEigenVector(validPoints, normal);
            }
            else if (validPoints.size() == 2) {
                cv::Vec3f v1 = validPoints[0];
                cv::Vec3f v2 = validPoints[1];
                normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
                normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
                normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
                success = (cv::norm(normal) > 1e-6);
            }

            if (success) {
                normalizeNormal(normal);
                if (normal[2] < 0) normal *= -1;
                normals.at<cv::Vec3f>(y, x) = normal;
            }
            else {
                normal = getNearestValidNormalWithDepthDiff(depthMap, x, y, centerDepth, rows, cols);
                normals.at<cv::Vec3f>(y, x) = normal;
            }
        }
    }
    return true;
}

// Visualize per-pixel normals in BGR by mapping each component [-1, 1] -> [0, 255].
void Phong::visualizeNormals(cv::Mat& normalVis) const
{
    if (normals.empty()) {
        std::cerr << "Warning: normals are empty." << std::endl;
        normalVis = cv::Mat();
        return;
    }
    normals.convertTo(normalVis, CV_8UC3, 127.5, 127.5);
}

// Phong shading: I = I_a + I_d + I_s
//   I_a = w_a * I_l
//   I_d = w_d * I_l * max(N . L, 0)
//   I_s = w_s * I_l * max(R . V, 0)^n,    R = 2 (N . L) N - L
// The final pixel is the base BGR color modulated by the clamped intensity.
bool Phong::renderPhong(const cv::Mat& baseColorBGR8,
    cv::Mat& shadedBGR8,
    cv::Vec3f lightDir,
    cv::Vec3f viewDir,
    float lightIntensity,  // I_l, overall illumination intensity
    float ambient,         // w_a, ambient weight
    float kd,              // w_d, diffuse weight
    float ks,              // w_s, specular weight
    float shininess,       // n, specular exponent
    const cv::Mat& validMask
) const
{
    if (normals.empty() || normals.type() != CV_32FC3) return false;
    if (baseColorBGR8.empty() || baseColorBGR8.type() != CV_8UC3) return false;
    if (baseColorBGR8.size() != normals.size()) return false;

    // Normalize light / view directions.
    float lenL = (float)cv::norm(lightDir);
    float lenV = (float)cv::norm(viewDir);
    if (lenL > 1e-6) lightDir /= lenL;
    if (lenV > 1e-6) viewDir  /= lenV;

    // Per-component channels of the normal map.
    std::vector<cv::Mat> n;
    cv::split(normals, n);
    cv::Mat nx = n[0], ny = n[1], nz = n[2];

    // Diffuse component: kd * I_l * max(N . L, 0).
    cv::Mat nDotL = nx * lightDir[0] + ny * lightDir[1] + nz * lightDir[2];
    cv::max(nDotL, 0.0f, nDotL);
    cv::Mat diffuse = kd * nDotL * lightIntensity;

    // Reflection direction R = 2 (N . L) N - L.
    cv::Mat twoNdL = 2.0f * nDotL;
    cv::Mat rx = twoNdL.mul(nx) - lightDir[0];
    cv::Mat ry = twoNdL.mul(ny) - lightDir[1];
    cv::Mat rz = twoNdL.mul(nz) - lightDir[2];

    // Normalize R so the dot product with V stays in [-1, 1].
    cv::Mat rNorm;
    cv::sqrt(rx.mul(rx) + ry.mul(ry) + rz.mul(rz), rNorm);
    cv::threshold(rNorm, rNorm, 1e-6f, 1.0f, cv::THRESH_BINARY);
    rx /= rNorm; ry /= rNorm; rz /= rNorm;

    cv::Mat rDotV = rx * viewDir[0] + ry * viewDir[1] + rz * viewDir[2];
    cv::max(rDotV, 0.0f, rDotV);

    cv::Mat spec;
    cv::pow(rDotV, shininess, spec);
    cv::Mat specular = ks * spec * lightIntensity;

    // Ambient term: a constant intensity field.
    cv::Mat ambientTerm = cv::Mat::ones(nx.size(), CV_32F) * ambient * lightIntensity;

    // Total intensity, clamped to [0, 1].
    cv::Mat out = ambientTerm + diffuse + specular;
    cv::min(out, 1.0f, out);

    // Modulate base color: I * C(x,y).
    cv::Mat f;
    baseColorBGR8.convertTo(f, CV_32FC3, 1.0f / 255.0f);

    cv::Mat mul;
    std::vector<cv::Mat> channels = { out, out, out };
    cv::merge(channels, mul);

    cv::Mat res = f.mul(mul);
    res.convertTo(shadedBGR8, CV_8UC3, 255.0f);

    // Force invalid pixels to black.
    cv::Mat mask;
    if (validMask.empty())
        mask = (cv::abs(nx) + cv::abs(ny) + cv::abs(nz)) > 0.1f;
    else
        mask = validMask.clone();

    shadedBGR8.setTo(0, ~mask);
    return true;
}
