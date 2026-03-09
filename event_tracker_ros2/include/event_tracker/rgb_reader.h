#ifndef EVENT_TRACKER__RGB_READER_H
#define EVENT_TRACKER__RGB_READER_H

// RGB frame reader with egomotion-compensated motion detection.
//
// Unchanged from the standalone project — pure OpenCV + HDF5 I/O.
// The ROS node wraps this to publish sensor_msgs/Image.

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

class RGBReader {
public:
    bool open(const std::string& path);

    // Open RGB frames from an H5 file (M3ED format: ovc/rgb/data + ovc/ts)
    bool openH5(const std::string& h5_path);

    // Get the frame closest to the given time (in seconds).
    bool getFrame(double t_sec, cv::Mat& frame);

    // Compute an egomotion-compensated motion mask.
    bool getMotionMask(double t_sec,
                       const Eigen::Matrix<double, 6, 1>& bg_affine,
                       double cx, double cy, double sx, double sy,
                       cv::Mat& mask);

    // Fallback: uncompensated motion mask
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

    bool h5_mode_ = false;
    std::string h5_path_;
    std::vector<double> h5_timestamps_;

    cv::Mat prev_gray_;
    int prev_frame_idx_ = -1;

    cv::Mat cached_bgr_;
    cv::Mat cached_gray_;
    int cached_frame_idx_ = -1;

    cv::Mat cached_mask_;
    int cached_mask_idx_ = -1;

    bool readFrameAt(int idx, cv::Mat& bgr, cv::Mat& gray);
    bool readH5FrameAt(int idx, cv::Mat& bgr, cv::Mat& gray);
    int findH5Frame(double t_sec) const;
    void resizeToTarget(cv::Mat& mask) const;
};

#endif // EVENT_TRACKER__RGB_READER_H
