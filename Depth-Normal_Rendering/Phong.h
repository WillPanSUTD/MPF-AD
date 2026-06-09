#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Surface-normal estimation + Phong shading for depth-driven rendering.
 *
 * Pipeline:
 *  1. computeNormals(): per-pixel surface normals from a 32-bit float depth map,
 *     via PCA on a 3D neighborhood (with sensible fallbacks at boundaries).
 *  2. renderPhong(): Phong shading (ambient + diffuse + specular) modulating a
 *     base BGR image (typically the LUT pseudocolor output) using those normals.
 *
 *  Auxiliary helpers (normalize, PCA solve, depth-scale, boundary fallback,
 *  visualization) are exposed so the pipeline can be debugged step by step.
 */
class Phong
{
public:
    // ============================== Members ==============================
    /**
     * Per-pixel unit surface normals (CV_32FC3). Each element is a
     * normalized (nx, ny, nz) triple; pixels with no valid neighborhood
     * are written as (0, 0, 0).
     */
    cv::Mat normals;

    // ============================ Helpers ===============================
    /**
     * In-place L2 normalization of a 3D vector. Vectors with magnitude
     * below 1e-6 are replaced with (0, 0, 1).
     */
    void normalizeNormal(cv::Vec3f& normal);

    /**
     * Estimate a plane normal from a set of 3D points via PCA: returns the
     * eigenvector of the covariance matrix corresponding to the smallest
     * eigenvalue. Returns false if the point set is too small (< 3 points)
     * or eigendecomposition fails.
     */
    bool computeEigenVector(const std::vector<cv::Vec3f>& points, cv::Vec3f& normal);

    /**
     * Auto-compute a Z-axis scale so depth differences (mm) and pixel
     * offsets (px) live in comparable ranges. Returns 512 / (d_max - d_min)
     * for valid depth ranges, or 1.0 as a safe fallback.
     */
    float computeDepthScale(const cv::Mat& depth32F);

    /**
     * Boundary fallback: when the central pixel has fewer than two valid
     * neighbors, take the surface normal of the nearby pixel with the
     * largest depth difference (within a 1-pixel search radius).
     */
    cv::Vec3f getNearestValidNormalWithDepthDiff(const cv::Mat& depthMap, int x, int y,
                                                 float centerDepth, int rows, int cols);

    // ============================ Main API ===============================
    /**
     * Estimate per-pixel surface normals from a depth map.
     * @param depthMap        single-channel CV_32FC1 depth map.
     * @param use8Neighbors   if true, sample 8 neighbors; otherwise 4.
     * @return true on success.
     */
    bool computeNormals(const cv::Mat& depthMap, bool use8Neighbors = true);

    /**
     * Render the per-pixel normals as a BGR visualization image, mapping
     * each normal component from [-1, 1] to [0, 255].
     */
    void visualizeNormals(cv::Mat& normalVis) const;

    /**
     * Accessor.
     */
    cv::Mat getNormals() const { return normals; }

    /**
     * Phong shading: ambient + diffuse + specular modulating a base BGR
     * image. All vectors are in the same image-space coordinate system.
     *
     * @param baseColorBGR8     Base BGR image (e.g. LUT pseudocolor), CV_8UC3.
     * @param shadedBGR8        Output rendered image, CV_8UC3.
     * @param lightDir          Light source direction (normalized internally).
     * @param viewDir           View direction (normalized internally).
     * @param lightIntensity    Overall light intensity I_l.
     * @param ambient           Ambient weight w_a in [0, 1].
     * @param kd                Diffuse weight w_d in [0, 1].
     * @param ks                Specular weight w_s in [0, 1].
     * @param shininess         Specular exponent n (Phong power).
     * @param validMask         Optional mask of valid pixels (CV_8U); leave
     *                          empty to derive it from non-zero normals.
     * @return true on success.
     */
    bool renderPhong(const cv::Mat& baseColorBGR8,
        cv::Mat& shadedBGR8,
        cv::Vec3f lightDir,
        cv::Vec3f viewDir,
        float lightIntensity,
        float ambient,
        float kd,
        float ks,
        float shininess,
        const cv::Mat& validMask
    ) const;
};
