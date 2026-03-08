#ifndef RGB_READER_H
#define RGB_READER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

// RGB frame reader with egomotion-compensated motion detection.
//
// On a moving camera, simple frame differencing flags the whole scene as
// "moving". To get around this, we warp the previous frame using the
// estimated background (camera) motion from CMax, then difference. Static
// stuff cancels out, leaving only genuinely moving objects.
//
// The warp uses an affine velocity field from the CMax motion model:
//   vx(x,y) = a*xn + b*yn + tx
//   vy(x,y) = c*xn + d*yn + ty
// where xn, yn are normalized to [-1, 1].
//
// Two backends:
//   1. AVI/video file via OpenCV VideoCapture
//   2. HDF5 embedded RGB frames (M3ED format: ovc/rgb/data + ovc/ts)
class RGBReader {
public:
    bool open(const std::string& path);

    // Open RGB frames from an H5 file (M3ED format: ovc/rgb/data + ovc/ts)
    bool openH5(const std::string& h5_path);

    // Get the frame closest to the given time (in seconds).
    bool getFrame(double t_sec, cv::Mat& frame);

    // Compute an egomotion-compensated motion mask.
    // bg_affine: [a, b, c, d, tx, ty] — the BG affine velocity field
    // The image center and normalisation are provided for the warp model.
    bool getMotionMask(double t_sec,
                       const Eigen::Matrix<double, 6, 1>& bg_affine,
                       double cx, double cy, double sx, double sy,
                       cv::Mat& mask);

    // Fallback: uncompensated motion mask (for when no BG affine available yet)
    bool getMotionMaskSimple(double t_sec, cv::Mat& mask);

    int width()  const { return width_;  }
    int height() const { return height_; }
    double fps() const { return fps_; }
    double duration() const { return duration_; }
    int numFrames() const { return total_frames_; }
    double startTime() const { return h5_timestamps_.empty() ? 0.0 : h5_timestamps_.front(); }
    bool isOpen() const { return cap_.isOpened() || h5_mode_; }

    void setTargetSize(int w, int h) { target_w_ = w; target_h_ = h; }

private:
    cv::VideoCapture cap_;
    int width_  = 0;
    int height_ = 0;
    double fps_ = 0;
    double duration_ = 0;
    int total_frames_ = 0;

    int target_w_ = 0;
    int target_h_ = 0;

    // H5 mode: frames stored in memory after reading from HDF5
    bool h5_mode_ = false;
    std::string h5_path_;
    std::vector<double> h5_timestamps_;   // frame timestamps in seconds (from event t0)

    // Cache previous frame for differencing
    cv::Mat prev_gray_;
    int prev_frame_idx_ = -1;

    // Cache current frame to avoid re-reading for same index
    cv::Mat cached_bgr_;
    cv::Mat cached_gray_;
    int cached_frame_idx_ = -1;

    // Cache the last computed motion mask and its frame index
    cv::Mat cached_mask_;
    int cached_mask_idx_ = -1;

    // Read a specific frame index, handle BGRA conversion
    bool readFrameAt(int idx, cv::Mat& bgr, cv::Mat& gray);

    // Read a frame from HDF5 file by index
    bool readH5FrameAt(int idx, cv::Mat& bgr, cv::Mat& gray);

    // Find the H5 frame index closest to a given time
    int findH5Frame(double t_sec) const;

    // Resize mask to target resolution if needed
    void resizeToTarget(cv::Mat& mask) const;
};

#endif // RGB_READER_H
