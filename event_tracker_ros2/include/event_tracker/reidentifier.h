#ifndef EVENT_TRACKER__REIDENTIFIER_H
#define EVENT_TRACKER__REIDENTIFIER_H

// Target re-identification and motion prediction.
//
// Unchanged from the standalone project — pure OpenCV, no ROS deps.
//
// 1. **Kalman filter** — maintains a position+velocity state for the bbox
//    center. Fed with MOSSE and event observations each frame. When MOSSE
//    loses the target, the Kalman prediction tells us where to look.
//
// 2. **Template matching re-ID** — stores a small gallery of recent target
//    patches (grabbed when tracking confidence is high). When MOSSE fails,
//    we search a neighborhood around the Kalman prediction using normalized
//    cross-correlation. If the best match exceeds a threshold, we re-init.
//
// The tracker calls update() every frame when MOSSE is happy, and calls
// recover() when MOSSE reports failure. recover() returns true if it found
// the target again, along with the new bbox to re-init MOSSE with.

#include <opencv2/core.hpp>
#include <deque>

class ReIdentifier {
public:
    struct Config {
        // Kalman filter tuning
        double process_noise   = 80.0;
        double measurement_noise = 15.0;

        // Template gallery
        int    gallery_size    = 5;
        int    gallery_update_interval = 8;

        // Re-ID search
        double search_radius_factor = 1.5;
        double match_threshold = 0.55;
        int    max_lost_frames = 60;
        int    recover_interval = 3;

        // Scale adaptation
        double scale_step = 0.05;

        // Drift detection
        double drift_edge_fraction = 0.60;
        double drift_jump_threshold = 120.0;
        int    cooldown_after_reid = 15;
    };

    ReIdentifier();
    explicit ReIdentifier(const Config& cfg);

    void init(const cv::Rect2d& bbox, const cv::Mat& frame);

    cv::Rect2d update(const cv::Rect2d& mosse_bbox,
                      const cv::Point2d& event_shift,
                      double event_confidence,
                      const cv::Mat& frame);

    bool hasDrifted(const cv::Rect2d& mosse_bbox, const cv::Mat& frame,
                    int frame_w, int frame_h) const;

    bool recover(const cv::Mat& frame, cv::Rect2d& recovered_bbox);

    int lostFrames() const { return lost_frames_; }
    bool gaveUp() const { return lost_frames_ > cfg_.max_lost_frames; }
    cv::Rect2d predictedBBox() const;
    cv::Point2d velocity() const;
    double lastAppearanceScore() const { return last_appearance_score_; }

private:
    Config cfg_;

    // Kalman filter state: [cx, cy, vx, vy]
    cv::Mat kf_state_;
    cv::Mat kf_cov_;
    cv::Mat kf_F_;
    cv::Mat kf_Q_;
    cv::Mat kf_H_;
    cv::Mat kf_R_;
    bool    kf_initialized_ = false;

    void kf_predict();
    void kf_correct(double cx, double cy);

    double bbox_w_ = 0, bbox_h_ = 0;

    struct TemplateEntry {
        cv::Mat patch;
        cv::Mat color_hist;
    };
    std::deque<TemplateEntry> gallery_;
    int frames_since_gallery_update_ = 0;

    void saveTemplate(const cv::Rect2d& bbox, const cv::Mat& frame);
    cv::Mat extractPatch(const cv::Rect2d& bbox, const cv::Mat& frame) const;
    cv::Mat computeColorHist(const cv::Rect2d& bbox, const cv::Mat& frame) const;

    double last_appearance_score_ = 1.0;
    cv::Rect2d prev_bbox_;
    bool has_prev_bbox_ = false;
    int cooldown_ = 0;

    double measureAppearance(const cv::Rect2d& bbox, const cv::Mat& frame) const;

    int lost_frames_ = 0;
    double searchAtScale(const cv::Mat& frame, const cv::Rect2d& search_bbox,
                         double scale, cv::Point2d& best_offset) const;

    static constexpr int kPatchW = 64;
    static constexpr int kPatchH = 64;
};

#endif // EVENT_TRACKER__REIDENTIFIER_H
