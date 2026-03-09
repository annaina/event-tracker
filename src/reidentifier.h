#ifndef REIDENTIFIER_H
#define REIDENTIFIER_H

#include <opencv2/core.hpp>
#include <deque>

// Target re-identification and motion prediction.
//
// This module does two things:
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

class ReIdentifier {
public:
    struct Config {
        // Kalman filter tuning
        double process_noise   = 80.0;   // how much we trust the motion model
        double measurement_noise = 15.0; // how much we trust MOSSE position

        // Template gallery
        int    gallery_size    = 5;       // how many patches to keep
        int    gallery_update_interval = 8; // frames between gallery snapshots

        // Re-ID search
        double search_radius_factor = 1.5; // search radius = factor * bbox diagonal
        double match_threshold = 0.55;     // NCC threshold to accept a re-ID (high to avoid false matches)
        int    max_lost_frames = 60;       // give up after this many frames lost
        int    recover_interval = 3;       // only attempt re-ID every N lost frames

        // Scale adaptation
        double scale_step = 0.05;          // try scales at 1 ± step

        // Drift detection — keep it simple. NCC-based drift was too
        // trigger-happy and caused feedback loops, so we only use
        // out-of-frame and jump detection now.
        double drift_edge_fraction = 0.60; // bbox this far off-screen = gone
        double drift_jump_threshold = 120.0;// sudden jump threshold (px)
        int    cooldown_after_reid = 15;   // don't check drift for N frames after re-ID
    };

    ReIdentifier();
    explicit ReIdentifier(const Config& cfg);

    // Call this once when the user selects the initial bbox.
    void init(const cv::Rect2d& bbox, const cv::Mat& frame);

    // Call every frame when MOSSE tracking succeeds.
    // Updates the Kalman filter and occasionally saves a template.
    // Returns the Kalman-smoothed bbox (fused MOSSE + event prediction).
    cv::Rect2d update(const cv::Rect2d& mosse_bbox,
                      const cv::Point2d& event_shift,
                      double event_confidence,
                      const cv::Mat& frame);

    // Check whether MOSSE has drifted away from the real target.
    // Call this every frame *before* deciding to trust MOSSE.
    // Returns true if the current bbox looks wrong (should trigger re-ID).
    bool hasDrifted(const cv::Rect2d& mosse_bbox, const cv::Mat& frame,
                    int frame_w, int frame_h) const;

    // Call when MOSSE reports tracking failure or drift is detected.
    // Tries to re-identify the target near the Kalman prediction.
    // Returns true if re-ID succeeded, and fills out recovered_bbox.
    bool recover(const cv::Mat& frame, cv::Rect2d& recovered_bbox);

    // How many consecutive frames we've been lost.
    int lostFrames() const { return lost_frames_; }

    // Whether we've given up (lost too long).
    bool gaveUp() const { return lost_frames_ > cfg_.max_lost_frames; }

    // The Kalman-predicted bbox (useful for drawing even when lost).
    cv::Rect2d predictedBBox() const;

    // Current estimated velocity in pixels/frame.
    cv::Point2d velocity() const;

    // Last measured appearance similarity (0-1, higher = better match)
    double lastAppearanceScore() const { return last_appearance_score_; }

private:
    Config cfg_;

    // -- Kalman filter state --
    //
    // State: [cx, cy, vx, vy]  (bbox center position + velocity)
    // We track the center and keep width/height separately.
    cv::Mat kf_state_;       // 4x1
    cv::Mat kf_cov_;         // 4x4
    cv::Mat kf_F_;           // 4x4 transition
    cv::Mat kf_Q_;           // 4x4 process noise
    cv::Mat kf_H_;           // 2x4 measurement
    cv::Mat kf_R_;           // 2x2 measurement noise
    bool    kf_initialized_ = false;

    void kf_predict();
    void kf_correct(double cx, double cy);

    // -- Bbox size tracking (simple EMA) --
    double bbox_w_ = 0, bbox_h_ = 0;

    // -- Template gallery --
    struct TemplateEntry {
        cv::Mat patch;       // grayscale, resized to canonical size
        cv::Mat color_hist;  // tiny color histogram for quick rejection
    };
    std::deque<TemplateEntry> gallery_;
    int frames_since_gallery_update_ = 0;

    void saveTemplate(const cv::Rect2d& bbox, const cv::Mat& frame);
    cv::Mat extractPatch(const cv::Rect2d& bbox, const cv::Mat& frame) const;
    cv::Mat computeColorHist(const cv::Rect2d& bbox, const cv::Mat& frame) const;

    // -- Drift detection state --
    double last_appearance_score_ = 1.0;
    cv::Rect2d prev_bbox_;           // previous frame's bbox, for jump detection
    bool has_prev_bbox_ = false;
    int cooldown_ = 0;               // frames remaining before drift checks resume

    double measureAppearance(const cv::Rect2d& bbox, const cv::Mat& frame) const;

    // -- Re-ID search --
    int lost_frames_ = 0;
    double searchAtScale(const cv::Mat& frame, const cv::Rect2d& search_bbox,
                         double scale, cv::Point2d& best_offset) const;

    // Canonical patch size for template matching
    static constexpr int kPatchW = 64;
    static constexpr int kPatchH = 64;
};

#endif // REIDENTIFIER_H
