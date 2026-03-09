// Re-identification + Kalman filter + drift detection module.
// Identical to standalone version — only include path changed.

#include "event_tracker/reidentifier.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

ReIdentifier::ReIdentifier() : ReIdentifier(Config{}) {}

ReIdentifier::ReIdentifier(const Config& cfg) : cfg_(cfg) {
    kf_state_ = cv::Mat::zeros(4, 1, CV_64F);
    kf_cov_   = cv::Mat::eye(4, 4, CV_64F) * 1000.0;

    kf_F_ = cv::Mat::eye(4, 4, CV_64F);
    kf_F_.at<double>(0, 2) = 1.0;
    kf_F_.at<double>(1, 3) = 1.0;

    kf_Q_ = cv::Mat::zeros(4, 4, CV_64F);
    double q = cfg_.process_noise;
    kf_Q_.at<double>(0, 0) = q * 0.25;
    kf_Q_.at<double>(1, 1) = q * 0.25;
    kf_Q_.at<double>(2, 2) = q;
    kf_Q_.at<double>(3, 3) = q;

    kf_H_ = cv::Mat::zeros(2, 4, CV_64F);
    kf_H_.at<double>(0, 0) = 1.0;
    kf_H_.at<double>(1, 1) = 1.0;

    kf_R_ = cv::Mat::eye(2, 2, CV_64F) * cfg_.measurement_noise;
}

// ---------------------------------------------------------------------------
//  Kalman internals
// ---------------------------------------------------------------------------

void ReIdentifier::kf_predict() {
    kf_state_ = kf_F_ * kf_state_;
    kf_cov_ = kf_F_ * kf_cov_ * kf_F_.t() + kf_Q_;
}

void ReIdentifier::kf_correct(double cx, double cy) {
    cv::Mat z = (cv::Mat_<double>(2, 1) << cx, cy);
    cv::Mat y = z - kf_H_ * kf_state_;
    cv::Mat S = kf_H_ * kf_cov_ * kf_H_.t() + kf_R_;
    cv::Mat K = kf_cov_ * kf_H_.t() * S.inv();
    kf_state_ = kf_state_ + K * y;
    cv::Mat I = cv::Mat::eye(4, 4, CV_64F);
    kf_cov_ = (I - K * kf_H_) * kf_cov_;
}

// ---------------------------------------------------------------------------
//  Template gallery helpers
// ---------------------------------------------------------------------------

cv::Mat ReIdentifier::extractPatch(const cv::Rect2d& bbox, const cv::Mat& frame) const {
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
//  init
// ---------------------------------------------------------------------------

void ReIdentifier::init(const cv::Rect2d& bbox, const cv::Mat& frame) {
    double cx = bbox.x + bbox.width / 2.0;
    double cy = bbox.y + bbox.height / 2.0;

    kf_state_.at<double>(0) = cx;
    kf_state_.at<double>(1) = cy;
    kf_state_.at<double>(2) = 0;
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
//  measureAppearance
// ---------------------------------------------------------------------------

double ReIdentifier::measureAppearance(const cv::Rect2d& bbox,
                                        const cv::Mat& frame) const {
    if (gallery_.empty()) return 1.0;

    cv::Mat patch = extractPatch(bbox, frame);
    if (patch.empty()) return 0.0;

    const auto& ref = gallery_.back();
    cv::Mat result;
    cv::matchTemplate(patch, ref.patch, result, cv::TM_CCOEFF_NORMED);
    return result.at<float>(0, 0);
}

// ---------------------------------------------------------------------------
//  hasDrifted
// ---------------------------------------------------------------------------

bool ReIdentifier::hasDrifted(const cv::Rect2d& mosse_bbox,
                               const cv::Mat& frame,
                               int frame_w, int frame_h) const {
    auto* self = const_cast<ReIdentifier*>(this);

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
//  update
// ---------------------------------------------------------------------------

cv::Rect2d ReIdentifier::update(const cv::Rect2d& mosse_bbox,
                                 const cv::Point2d& event_shift,
                                 double event_confidence,
                                 const cv::Mat& frame) {
    lost_frames_ = 0;

    kf_predict();

    double mosse_cx = mosse_bbox.x + mosse_bbox.width / 2.0;
    double mosse_cy = mosse_bbox.y + mosse_bbox.height / 2.0;

    double cx = mosse_cx;
    double cy = mosse_cy;
    if (event_confidence > 0.3) {
        double w = std::min(0.3, event_confidence * 0.4);
        cx = mosse_cx * (1.0 - w) + (mosse_cx + event_shift.x) * w;
        cy = mosse_cy * (1.0 - w) + (mosse_cy + event_shift.y) * w;
    }

    kf_correct(cx, cy);

    double alpha_size = 0.1;
    bbox_w_ = alpha_size * mosse_bbox.width  + (1.0 - alpha_size) * bbox_w_;
    bbox_h_ = alpha_size * mosse_bbox.height + (1.0 - alpha_size) * bbox_h_;

    frames_since_gallery_update_++;
    if (frames_since_gallery_update_ >= cfg_.gallery_update_interval) {
        last_appearance_score_ = measureAppearance(mosse_bbox, frame);
    }
    if (frames_since_gallery_update_ >= cfg_.gallery_update_interval
        && last_appearance_score_ > 0.40) {
        saveTemplate(mosse_bbox, frame);
        frames_since_gallery_update_ = 0;
    }

    prev_bbox_ = mosse_bbox;
    has_prev_bbox_ = true;

    return predictedBBox();
}

// ---------------------------------------------------------------------------
//  recover
// ---------------------------------------------------------------------------

bool ReIdentifier::recover(const cv::Mat& frame, cv::Rect2d& recovered_bbox) {
    lost_frames_++;

    if (gallery_.empty() || !kf_initialized_) return false;
    if (gaveUp()) return false;

    kf_predict();

    kf_state_.at<double>(2) *= 0.92;
    kf_state_.at<double>(3) *= 0.92;

    cv::Rect2d pred = predictedBBox();

    double vis_x0 = std::max(0.0, pred.x);
    double vis_y0 = std::max(0.0, pred.y);
    double vis_x1 = std::min((double)frame.cols, pred.x + pred.width);
    double vis_y1 = std::min((double)frame.rows, pred.y + pred.height);
    double vis_area = std::max(0.0, vis_x1 - vis_x0) * std::max(0.0, vis_y1 - vis_y0);
    double tot_area = pred.width * pred.height;
    if (tot_area <= 0 || vis_area / tot_area < 0.75) {
        return false;
    }

    double pred_cx = pred.x + pred.width / 2.0;
    double pred_cy = pred.y + pred.height / 2.0;
    pred_cx = std::max(pred.width / 2.0, std::min((double)frame.cols - pred.width / 2.0, pred_cx));
    pred_cy = std::max(pred.height / 2.0, std::min((double)frame.rows - pred.height / 2.0, pred_cy));
    pred.x = pred_cx - pred.width / 2.0;
    pred.y = pred_cy - pred.height / 2.0;

    if (lost_frames_ % cfg_.recover_interval != 1 && lost_frames_ > 1)
        return false;

    cv::Point2d offset;
    double score = searchAtScale(frame, pred, 1.0, offset);

    if (score < cfg_.match_threshold) {
        cv::Point2d offset2;
        double score2 = searchAtScale(frame, pred, 1.0 + cfg_.scale_step, offset2);
        if (score2 > score) { score = score2; offset = offset2; }
    }

    if (score >= cfg_.match_threshold) {
        cv::Rect2d found(
            pred.x + offset.x, pred.y + offset.y,
            pred.width, pred.height);

        found.x = std::max(0.0, std::min((double)frame.cols - found.width, found.x));
        found.y = std::max(0.0, std::min((double)frame.rows - found.height, found.y));

        double cx = found.x + found.width / 2.0;
        double cy = found.y + found.height / 2.0;
        kf_correct(cx, cy);

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
//  searchAtScale
// ---------------------------------------------------------------------------

double ReIdentifier::searchAtScale(const cv::Mat& frame,
                                    const cv::Rect2d& pred,
                                    double scale,
                                    cv::Point2d& best_offset) const {
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
