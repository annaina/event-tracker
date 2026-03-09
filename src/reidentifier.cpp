// Re-identification + Kalman filter + drift detection module.
//
// The key insight: MOSSE almost never says "I lost it." It just silently
// drifts onto whatever texture is nearby. So we can't rely on MOSSE's own
// success/failure flag. Instead, we actively check every frame whether
// the current bbox still looks like our target (NCC against the gallery,
// color histogram check, out-of-frame detection, sudden jumps).
//
// When drift is detected, we switch to recovery mode: the Kalman filter
// keeps predicting where the target should be, and we search around that
// prediction with template matching. If we get a good enough match,
// we re-init MOSSE there and keep going.

#include "reidentifier.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

ReIdentifier::ReIdentifier() : ReIdentifier(Config{}) {}

ReIdentifier::ReIdentifier(const Config& cfg) : cfg_(cfg) {
    // Kalman state: [cx, cy, vx, vy]
    kf_state_ = cv::Mat::zeros(4, 1, CV_64F);
    kf_cov_   = cv::Mat::eye(4, 4, CV_64F) * 1000.0;  // start uncertain

    // Constant-velocity transition: x' = x + vx, y' = y + vy
    kf_F_ = cv::Mat::eye(4, 4, CV_64F);
    kf_F_.at<double>(0, 2) = 1.0;   // cx += vx
    kf_F_.at<double>(1, 3) = 1.0;   // cy += vy

    // Process noise — how much we expect the target to deviate from
    // constant velocity each frame. Higher = more responsive, noisier.
    kf_Q_ = cv::Mat::zeros(4, 4, CV_64F);
    double q = cfg_.process_noise;
    kf_Q_.at<double>(0, 0) = q * 0.25;
    kf_Q_.at<double>(1, 1) = q * 0.25;
    kf_Q_.at<double>(2, 2) = q;
    kf_Q_.at<double>(3, 3) = q;

    // Measurement matrix: we observe [cx, cy]
    kf_H_ = cv::Mat::zeros(2, 4, CV_64F);
    kf_H_.at<double>(0, 0) = 1.0;
    kf_H_.at<double>(1, 1) = 1.0;

    // Measurement noise
    kf_R_ = cv::Mat::eye(2, 2, CV_64F) * cfg_.measurement_noise;
}

// ---------------------------------------------------------------------------
//  Kalman internals
// ---------------------------------------------------------------------------

void ReIdentifier::kf_predict() {
    // x' = F * x
    kf_state_ = kf_F_ * kf_state_;
    // P' = F * P * F^T + Q
    kf_cov_ = kf_F_ * kf_cov_ * kf_F_.t() + kf_Q_;
}

void ReIdentifier::kf_correct(double cx, double cy) {
    // Innovation: z - H*x
    cv::Mat z = (cv::Mat_<double>(2, 1) << cx, cy);
    cv::Mat y = z - kf_H_ * kf_state_;

    // Innovation covariance: S = H*P*H^T + R
    cv::Mat S = kf_H_ * kf_cov_ * kf_H_.t() + kf_R_;

    // Kalman gain: K = P * H^T * S^-1
    cv::Mat K = kf_cov_ * kf_H_.t() * S.inv();

    // Update state and covariance
    kf_state_ = kf_state_ + K * y;
    cv::Mat I = cv::Mat::eye(4, 4, CV_64F);
    kf_cov_ = (I - K * kf_H_) * kf_cov_;
}

// ---------------------------------------------------------------------------
//  Template gallery helpers
// ---------------------------------------------------------------------------

cv::Mat ReIdentifier::extractPatch(const cv::Rect2d& bbox, const cv::Mat& frame) const {
    // Clamp the bbox to the frame
    int x0 = std::max(0, (int)bbox.x);
    int y0 = std::max(0, (int)bbox.y);
    int x1 = std::min(frame.cols, (int)(bbox.x + bbox.width));
    int y1 = std::min(frame.rows, (int)(bbox.y + bbox.height));
    if (x1 <= x0 || y1 <= y0) return cv::Mat();

    cv::Mat roi = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat gray;
    if (roi.channels() == 3)
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roi.clone();

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(kPatchW, kPatchH), 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat ReIdentifier::computeColorHist(const cv::Rect2d& bbox, const cv::Mat& frame) const {
    // Quick 16-bin hue histogram for color consistency checking.
    // This lets us cheaply reject matches that have totally wrong color.
    int x0 = std::max(0, (int)bbox.x);
    int y0 = std::max(0, (int)bbox.y);
    int x1 = std::min(frame.cols, (int)(bbox.x + bbox.width));
    int y1 = std::min(frame.rows, (int)(bbox.y + bbox.height));
    if (x1 <= x0 || y1 <= y0) return cv::Mat();

    cv::Mat roi = frame(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int hbins = 16;
    int histSize[] = {hbins};
    float hrange[] = {0, 180};
    const float* ranges[] = {hrange};
    int channels[] = {0};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    cv::normalize(hist, hist, 1.0, 0, cv::NORM_L1);
    return hist;
}

void ReIdentifier::saveTemplate(const cv::Rect2d& bbox, const cv::Mat& frame) {
    cv::Mat patch = extractPatch(bbox, frame);
    if (patch.empty()) return;

    cv::Mat hist = computeColorHist(bbox, frame);

    gallery_.push_back({patch, hist});
    while ((int)gallery_.size() > cfg_.gallery_size)
        gallery_.pop_front();
}

// ---------------------------------------------------------------------------
//  init — called once when the user draws the initial bbox
// ---------------------------------------------------------------------------

void ReIdentifier::init(const cv::Rect2d& bbox, const cv::Mat& frame) {
    double cx = bbox.x + bbox.width / 2.0;
    double cy = bbox.y + bbox.height / 2.0;

    kf_state_.at<double>(0) = cx;
    kf_state_.at<double>(1) = cy;
    kf_state_.at<double>(2) = 0;   // no velocity yet
    kf_state_.at<double>(3) = 0;
    kf_cov_ = cv::Mat::eye(4, 4, CV_64F) * 100.0;
    kf_initialized_ = true;

    bbox_w_ = bbox.width;
    bbox_h_ = bbox.height;

    gallery_.clear();
    saveTemplate(bbox, frame);
    frames_since_gallery_update_ = 0;
    lost_frames_ = 0;
    last_appearance_score_ = 1.0;
    prev_bbox_ = bbox;
    has_prev_bbox_ = true;
    cooldown_ = cfg_.cooldown_after_reid;
}

// ---------------------------------------------------------------------------
//  measureAppearance — quick NCC against the most recent gallery entry only
// ---------------------------------------------------------------------------

double ReIdentifier::measureAppearance(const cv::Rect2d& bbox,
                                        const cv::Mat& frame) const {
    if (gallery_.empty()) return 1.0;

    cv::Mat patch = extractPatch(bbox, frame);
    if (patch.empty()) return 0.0;

    // Only compare against the most recent entry — it's the most relevant
    // and avoids looping over the whole gallery every frame.
    const auto& ref = gallery_.back();
    cv::Mat result;
    cv::matchTemplate(patch, ref.patch, result, cv::TM_CCOEFF_NORMED);
    return result.at<float>(0, 0);
}

// ---------------------------------------------------------------------------
//  hasDrifted — simple, hard-to-fake checks only
//
//  We learned the hard way that NCC-based drift detection causes feedback
//  loops: detect drift → re-ID matches random patch → drift detected again
//  next frame → repeat. So now we only use two checks:
//    1. Out-of-frame (most of the bbox is off-screen)
//    2. Sudden teleport (bbox jumped way too far in one frame)
//
//  Plus a cooldown: after re-ID or init, we skip checks for N frames so
//  MOSSE has time to settle onto whatever it's been re-initialized on.
// ---------------------------------------------------------------------------

bool ReIdentifier::hasDrifted(const cv::Rect2d& mosse_bbox,
                               const cv::Mat& frame,
                               int frame_w, int frame_h) const {
    auto* self = const_cast<ReIdentifier*>(this);

    // Cooldown: don't check drift right after re-ID or init
    if (self->cooldown_ > 0) {
        self->cooldown_--;
        return false;
    }

    // 1) Out-of-frame
    double vis_x0 = std::max(0.0, mosse_bbox.x);
    double vis_y0 = std::max(0.0, mosse_bbox.y);
    double vis_x1 = std::min((double)frame_w, mosse_bbox.x + mosse_bbox.width);
    double vis_y1 = std::min((double)frame_h, mosse_bbox.y + mosse_bbox.height);
    double vis_area = std::max(0.0, vis_x1 - vis_x0) * std::max(0.0, vis_y1 - vis_y0);
    double tot_area = mosse_bbox.width * mosse_bbox.height;

    if (tot_area > 0 && vis_area / tot_area < (1.0 - cfg_.drift_edge_fraction)) {
        std::cout << "[Drift] off-screen (" << (int)(100*vis_area/tot_area) << "% visible)\n";
        return true;
    }

    // 2) Sudden jump
    if (has_prev_bbox_) {
        double dx = (mosse_bbox.x + mosse_bbox.width/2) -
                    (prev_bbox_.x + prev_bbox_.width/2);
        double dy = (mosse_bbox.y + mosse_bbox.height/2) -
                    (prev_bbox_.y + prev_bbox_.height/2);
        double jump = std::sqrt(dx*dx + dy*dy);
        if (jump > cfg_.drift_jump_threshold) {
            std::cout << "[Drift] jumped " << (int)jump << "px\n";
            return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
//  update — called every frame when MOSSE is tracking successfully
// ---------------------------------------------------------------------------

cv::Rect2d ReIdentifier::update(const cv::Rect2d& mosse_bbox,
                                 const cv::Point2d& event_shift,
                                 double event_confidence,
                                 const cv::Mat& frame) {
    lost_frames_ = 0;

    // Predict forward from last state
    kf_predict();

    // The MOSSE bbox center is our primary measurement
    double mosse_cx = mosse_bbox.x + mosse_bbox.width / 2.0;
    double mosse_cy = mosse_bbox.y + mosse_bbox.height / 2.0;

    // If events are confident, blend the event-shifted position in.
    double cx = mosse_cx;
    double cy = mosse_cy;
    if (event_confidence > 0.3) {
        double w = std::min(0.3, event_confidence * 0.4);
        cx = mosse_cx * (1.0 - w) + (mosse_cx + event_shift.x) * w;
        cy = mosse_cy * (1.0 - w) + (mosse_cy + event_shift.y) * w;
    }

    kf_correct(cx, cy);

    // Smoothly adapt the bbox size
    double alpha_size = 0.1;
    bbox_w_ = alpha_size * mosse_bbox.width  + (1.0 - alpha_size) * bbox_w_;
    bbox_h_ = alpha_size * mosse_bbox.height + (1.0 - alpha_size) * bbox_h_;

    // Periodically measure appearance to keep last_appearance_score_
    // honest. We don't use this for drift detection (too trigger-happy),
    // but we do use it to gate gallery updates — no point saving a bad
    // patch as a template.
    frames_since_gallery_update_++;
    if (frames_since_gallery_update_ >= cfg_.gallery_update_interval) {
        last_appearance_score_ = measureAppearance(mosse_bbox, frame);
    }
    if (frames_since_gallery_update_ >= cfg_.gallery_update_interval
        && last_appearance_score_ > 0.40) {
        saveTemplate(mosse_bbox, frame);
        frames_since_gallery_update_ = 0;
    }

    // Remember this bbox for jump detection next frame
    prev_bbox_ = mosse_bbox;
    has_prev_bbox_ = true;

    return predictedBBox();
}

// ---------------------------------------------------------------------------
//  recover — called when MOSSE loses the target
// ---------------------------------------------------------------------------

bool ReIdentifier::recover(const cv::Mat& frame, cv::Rect2d& recovered_bbox) {
    lost_frames_++;

    if (gallery_.empty() || !kf_initialized_) return false;
    if (gaveUp()) return false;

    // Advance the Kalman prediction (constant-velocity coast).
    kf_predict();

    // Dampen the velocity while lost — the target probably changed
    // direction by now, so a stale velocity just sends the prediction
    // flying off-screen. Decay it toward zero.
    kf_state_.at<double>(2) *= 0.92;  // vx
    kf_state_.at<double>(3) *= 0.92;  // vy

    cv::Rect2d pred = predictedBBox();

    // KEY: Check visibility of the *unclamped* prediction. If the Kalman
    // thinks the target is mostly off-screen, DON'T search. The old code
    // clamped first and then checked — which always passed, so we'd search
    // near frame edges and match garbage. Now we only search when the
    // Kalman genuinely thinks the target is back in view.
    double vis_x0 = std::max(0.0, pred.x);
    double vis_y0 = std::max(0.0, pred.y);
    double vis_x1 = std::min((double)frame.cols, pred.x + pred.width);
    double vis_y1 = std::min((double)frame.rows, pred.y + pred.height);
    double vis_area = std::max(0.0, vis_x1 - vis_x0) * std::max(0.0, vis_y1 - vis_y0);
    double tot_area = pred.width * pred.height;
    if (tot_area <= 0 || vis_area / tot_area < 0.75) {
        // Target is still mostly off-screen. Just wait.
        return false;
    }

    // Now clamp for the actual search center.
    double pred_cx = pred.x + pred.width / 2.0;
    double pred_cy = pred.y + pred.height / 2.0;
    pred_cx = std::max(pred.width / 2.0, std::min((double)frame.cols - pred.width / 2.0, pred_cx));
    pred_cy = std::max(pred.height / 2.0, std::min((double)frame.rows - pred.height / 2.0, pred_cy));
    pred.x = pred_cx - pred.width / 2.0;
    pred.y = pred_cy - pred.height / 2.0;

    // Throttle: only run the expensive template search every N frames
    if (lost_frames_ % cfg_.recover_interval != 1 && lost_frames_ > 1)
        return false;

    // Single-scale search
    cv::Point2d offset;
    double score = searchAtScale(frame, pred, 1.0, offset);

    // If that didn't work well, try one more scale
    if (score < cfg_.match_threshold) {
        cv::Point2d offset2;
        double score2 = searchAtScale(frame, pred, 1.0 + cfg_.scale_step, offset2);
        if (score2 > score) { score = score2; offset = offset2; }
    }

    if (score >= cfg_.match_threshold) {
        cv::Rect2d found(
            pred.x + offset.x, pred.y + offset.y,
            pred.width, pred.height);

        // Clamp the found bbox to frame bounds
        found.x = std::max(0.0, std::min((double)frame.cols - found.width, found.x));
        found.y = std::max(0.0, std::min((double)frame.rows - found.height, found.y));

        // Correct the Kalman with the match position
        double cx = found.x + found.width / 2.0;
        double cy = found.y + found.height / 2.0;
        kf_correct(cx, cy);

        // Reset velocity on recovery — the old velocity is stale
        kf_state_.at<double>(2) = 0;
        kf_state_.at<double>(3) = 0;

        bbox_w_ = found.width;
        bbox_h_ = found.height;
        recovered_bbox = found;
        lost_frames_ = 0;
        cooldown_ = cfg_.cooldown_after_reid;

        std::cout << "[Re-ID] Recovered! NCC=" << score
                  << " at (" << (int)found.x << "," << (int)found.y << ")\n";
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
//  searchAtScale — slide the template across a search window
// ---------------------------------------------------------------------------

double ReIdentifier::searchAtScale(const cv::Mat& frame,
                                    const cv::Rect2d& pred,
                                    double scale,
                                    cv::Point2d& best_offset) const {
    // Tight search region around the Kalman prediction
    double diag = std::sqrt(pred.width * pred.width + pred.height * pred.height);
    double margin = diag * cfg_.search_radius_factor;

    int sx0 = std::max(0, (int)(pred.x - margin));
    int sy0 = std::max(0, (int)(pred.y - margin));
    int sx1 = std::min(frame.cols, (int)(pred.x + pred.width + margin));
    int sy1 = std::min(frame.rows, (int)(pred.y + pred.height + margin));
    if (sx1 - sx0 < 10 || sy1 - sy0 < 10) return -1;

    cv::Mat search_roi = frame(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
    cv::Mat search_gray;
    if (search_roi.channels() == 3)
        cv::cvtColor(search_roi, search_gray, cv::COLOR_BGR2GRAY);
    else
        search_gray = search_roi;

    int tw = std::max(8, (int)(pred.width * scale));
    int th = std::max(8, (int)(pred.height * scale));
    if (tw >= search_gray.cols || th >= search_gray.rows) return -1;

    // Only use the most recent gallery entry — it's the cheapest and
    // most relevant. The old approach of checking all entries was the
    // main reason recovery was so slow.
    const auto& entry = gallery_.back();
    cv::Mat tmpl;
    cv::resize(entry.patch, tmpl, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

    cv::Mat result;
    cv::matchTemplate(search_gray, tmpl, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    if (maxVal > 0) {
        double found_cx = sx0 + maxLoc.x + tw / 2.0;
        double found_cy = sy0 + maxLoc.y + th / 2.0;
        double pred_cx = pred.x + pred.width / 2.0;
        double pred_cy = pred.y + pred.height / 2.0;
        best_offset.x = found_cx - pred_cx;
        best_offset.y = found_cy - pred_cy;
    }

    return maxVal;
}

// ---------------------------------------------------------------------------
//  Accessors
// ---------------------------------------------------------------------------

cv::Rect2d ReIdentifier::predictedBBox() const {
    double cx = kf_state_.at<double>(0);
    double cy = kf_state_.at<double>(1);
    return cv::Rect2d(cx - bbox_w_ / 2.0, cy - bbox_h_ / 2.0, bbox_w_, bbox_h_);
}

cv::Point2d ReIdentifier::velocity() const {
    return {kf_state_.at<double>(2), kf_state_.at<double>(3)};
}
