// Event-Enhanced MOSSE Tracker — ROS 2 Node
//
// Translates the standalone tracker_main.cpp into a proper ROS 2 node.
// All tracking logic is identical. The difference is:
//   - Input comes from subscribed topics instead of direct H5 reading
//   - Output goes to published topics + visualization
//   - Parameters are ROS 2 params instead of CLI args
//
// Subscriptions:
//   ~/image_raw   (sensor_msgs/msg/Image)           — RGB frames
//   ~/events      (event_tracker/msg/EventArray)     — event batches
//
// Publications:
//   ~/tracking_status  (event_tracker/msg/TrackingStatus) — bbox + state
//   ~/debug_image      (sensor_msgs/msg/Image)            — annotated frame
//   ~/time_surface     (sensor_msgs/msg/Image)            — event visualization
//
// Parameters:
//   calib_path        — path to calibration.yaml
//   event_window_ms   — ms of events to use (default: 5.0)
//   show_overlay      — show event heatmap on RGB (default: true)
//   show_trail        — show trajectory trail (default: true)
//   reseat_threshold  — Kalman-reseat distance in px (default: 15.0)
//
// To start tracking, call the ~/init_bbox service with the bounding box,
// or use the interactive OpenCV window (same as standalone).

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include <deque>
#include <cmath>
#include <algorithm>

#include "event_tracker/reidentifier.h"
#include "event_tracker/auto_calibrate.h"
#include "event_tracker/msg/event_array.hpp"
#include "event_tracker/msg/tracking_status.hpp"

// ---------------------------------------------------------------------------
//  EventBBoxPredictor — identical to standalone
// ---------------------------------------------------------------------------

class EventBBoxPredictor {
public:
    EventBBoxPredictor(int img_w, int img_h)
        : img_w_(img_w), img_h_(img_h),
          time_surface_(img_h, img_w, CV_64FC1, cv::Scalar(0)),
          smooth_vx_(0), smooth_vy_(0) {}

    cv::Point2d predict(const std::vector<Event>& events, double batch_t0, double batch_t1,
                        const cv::Rect2d& bbox,
                        double& event_density, double& confidence) {
        if (events.empty() || bbox.width <= 0) {
            event_density = 0;
            confidence = 0;
            smooth_vx_ *= 0.5;
            smooth_vy_ *= 0.5;
            double dt = batch_t1 - batch_t0;
            return {smooth_vx_ * std::max(dt, 0.001), smooth_vy_ * std::max(dt, 0.001)};
        }

        int bx0 = std::max(1, (int)bbox.x);
        int by0 = std::max(1, (int)bbox.y);
        int bx1 = std::min(img_w_ - 2, (int)(bbox.x + bbox.width));
        int by1 = std::min(img_h_ - 2, (int)(bbox.y + bbox.height));

        double dt = batch_t1 - batch_t0;
        int total_in_bbox = 0;

        double sum_vx = 0, sum_vy = 0;
        int n_flow = 0;

        constexpr int R = 2;
        double recency = 0.030;

        for (const auto& e : events) {
            if (e.x >= 0 && e.x < img_w_ && e.y >= 0 && e.y < img_h_)
                time_surface_.at<double>(e.y, e.x) = e.t;

            if (e.x < bx0 || e.x > bx1 || e.y < by0 || e.y > by1)
                continue;
            total_in_bbox++;

            if (e.x < R || e.x >= img_w_ - R || e.y < R || e.y >= img_h_ - R)
                continue;

            double sum_xx = 0, sum_yy = 0, sum_xy = 0;
            double sum_xt = 0, sum_yt = 0;
            double sum_x = 0, sum_y = 0, sum_t = 0;
            int count = 0;

            for (int dy = -R; dy <= R; ++dy) {
                for (int dx = -R; dx <= R; ++dx) {
                    double ts = time_surface_.at<double>(e.y + dy, e.x + dx);
                    if (ts <= 0 || (e.t - ts) > recency) continue;
                    double lx = dx, ly = dy, lt = ts;
                    sum_x += lx; sum_y += ly; sum_t += lt;
                    sum_xx += lx*lx; sum_yy += ly*ly; sum_xy += lx*ly;
                    sum_xt += lx*lt; sum_yt += ly*lt;
                    count++;
                }
            }

            if (count < 6) continue;

            double n = count;
            double Sxx = sum_xx - sum_x*sum_x/n;
            double Syy = sum_yy - sum_y*sum_y/n;
            double Sxy = sum_xy - sum_x*sum_y/n;
            double Sxt = sum_xt - sum_x*sum_t/n;
            double Syt = sum_yt - sum_y*sum_t/n;

            double det = Sxx*Syy - Sxy*Sxy;
            if (std::abs(det) < 1e-20) continue;

            double a = (Syy*Sxt - Sxy*Syt) / det;
            double b = (Sxx*Syt - Sxy*Sxt) / det;

            double grad_sq = a*a + b*b;
            if (grad_sq < 1e-16) continue;

            double vx = -a / grad_sq;
            double vy = -b / grad_sq;

            double speed = std::sqrt(vx*vx + vy*vy);
            if (speed > 2000.0) continue;

            sum_vx += vx;
            sum_vy += vy;
            n_flow++;
        }

        event_density = (dt > 0) ? total_in_bbox / dt : 0;

        if (n_flow < 10) {
            double area = bbox.width * bbox.height;
            confidence = std::min(1.0, total_in_bbox / (area * 0.5));
            smooth_vx_ *= 0.7;
            smooth_vy_ *= 0.7;
            return {smooth_vx_ * dt, smooth_vy_ * dt};
        }

        double avg_vx = sum_vx / n_flow;
        double avg_vy = sum_vy / n_flow;

        constexpr double alpha = 0.3;
        smooth_vx_ = alpha * avg_vx + (1.0 - alpha) * smooth_vx_;
        smooth_vy_ = alpha * avg_vy + (1.0 - alpha) * smooth_vy_;

        double area = bbox.width * bbox.height;
        double density_ratio = total_in_bbox / area;
        confidence = std::min(1.0, density_ratio / 0.5);

        return {smooth_vx_ * dt, smooth_vy_ * dt};
    }

    cv::Mat renderTimeSurface(double t_now, double decay_sec = 0.05) const {
        cv::Mat vis(img_h_, img_w_, CV_8UC3, cv::Scalar(15, 15, 15));
        for (int y = 0; y < img_h_; ++y) {
            for (int x = 0; x < img_w_; ++x) {
                double ts = time_surface_.at<double>(y, x);
                if (ts <= 0) continue;
                double age = t_now - ts;
                if (age < decay_sec) {
                    int v = (int)(255.0 * (1.0 - age / decay_sec));
                    vis.at<cv::Vec3b>(y, x) = {0, (uchar)v, 0};
                }
            }
        }
        return vis;
    }

private:
    int img_w_, img_h_;
    cv::Mat time_surface_;
    double smooth_vx_, smooth_vy_;
};

// ---------------------------------------------------------------------------
//  BBoxTrail — identical to standalone
// ---------------------------------------------------------------------------

struct BBoxTrail {
    std::deque<cv::Rect2d> trail;
    size_t max_len = 30;

    void push(const cv::Rect2d& b) {
        trail.push_back(b);
        while (trail.size() > max_len) trail.pop_front();
    }
    void clear() { trail.clear(); }

    void draw(cv::Mat& img, const cv::Scalar& color) const {
        for (size_t i = 0; i < trail.size(); ++i) {
            double a = 0.1 + 0.9 * i / std::max((size_t)1, trail.size() - 1);
            cv::Scalar c = color * a;
            cv::Point center((int)(trail[i].x + trail[i].width/2),
                             (int)(trail[i].y + trail[i].height/2));
            cv::circle(img, center, 3, c, -1);
            if (i > 0) {
                cv::Point prev((int)(trail[i-1].x + trail[i-1].width/2),
                               (int)(trail[i-1].y + trail[i-1].height/2));
                cv::line(img, prev, center, c, 1, cv::LINE_AA);
            }
        }
    }
};

// ---------------------------------------------------------------------------
//  TrackerNode
// ---------------------------------------------------------------------------

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode() : Node("tracker_node") {
        // Parameters
        this->declare_parameter<std::string>("calib_path", "");
        this->declare_parameter<double>("event_window_ms", 5.0);
        this->declare_parameter<bool>("show_overlay", true);
        this->declare_parameter<bool>("show_trail", true);
        this->declare_parameter<bool>("gui", true);
        this->declare_parameter<double>("reseat_threshold", 15.0);

        calib_path_  = this->get_parameter("calib_path").as_string();
        evt_win_ms_  = this->get_parameter("event_window_ms").as_double();
        show_overlay_ = this->get_parameter("show_overlay").as_bool();
        show_trail_  = this->get_parameter("show_trail").as_bool();
        gui_         = this->get_parameter("gui").as_bool();
        reseat_threshold_ = this->get_parameter("reseat_threshold").as_double();

        // Load calibration
        if (!calib_path_.empty()) {
            CalibResult cal;
            if (loadCalibration(calib_path_, cal)) {
                sr_x_ = cal.sx; sr_y_ = cal.sy;
                off_x_ = cal.ox; off_y_ = cal.oy;
                RCLCPP_INFO(this->get_logger(),
                    "Loaded calibration: sx=%.4f sy=%.4f ox=%.1f oy=%.1f (score=%d%%)",
                    cal.sx, cal.sy, cal.ox, cal.oy, (int)(cal.score * 100));
            } else {
                RCLCPP_ERROR(this->get_logger(),
                    "Cannot load calibration: %s", calib_path_.c_str());
            }
        } else {
            // Fallback to M3ED intrinsics
            double fx_e = 1034.86, fy_e = 1033.48, cx_e = 629.70, cy_e = 357.60;
            double fx_r = 1268.56, fy_r = 1267.35, cx_r = 649.37, cy_r = 359.94;
            sr_x_  = fx_e / fx_r;
            sr_y_  = fy_e / fy_r;
            off_x_ = cx_e - cx_r * sr_x_;
            off_y_ = cy_e - cy_r * sr_y_;
            RCLCPP_WARN(this->get_logger(),
                "No calibration file — using hardcoded M3ED intrinsics. "
                "Set calib_path parameter.");
        }

        rs_x_  = 1.0 / sr_x_;
        rs_y_  = 1.0 / sr_y_;
        roff_x_ = -off_x_ / sr_x_;
        roff_y_ = -off_y_ / sr_y_;

        // Subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "~/image_raw", 10,
            std::bind(&TrackerNode::imageCallback, this, std::placeholders::_1));

        event_sub_ = this->create_subscription<event_tracker::msg::EventArray>(
            "~/events", 10,
            std::bind(&TrackerNode::eventCallback, this, std::placeholders::_1));

        // Publishers
        status_pub_ = this->create_publisher<event_tracker::msg::TrackingStatus>(
            "~/tracking_status", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "~/debug_image", 10);
        tsurf_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "~/time_surface", 10);

        RCLCPP_INFO(this->get_logger(),
            "Tracker node ready. Calibration: RGB->Evt scale=(%.4f,%.4f) offset=(%.1f,%.1f)",
            sr_x_, sr_y_, off_x_, off_y_);

        if (gui_) {
            RCLCPP_INFO(this->get_logger(),
                "GUI mode: SPACE=pause/select/track  R=reset  E=overlay  T=trail  Q=quit");
        }
    }

private:
    // --- Calibration ---
    double sr_x_ = 0.816, sr_y_ = 0.816;
    double off_x_ = 100.0, off_y_ = 64.1;
    double rs_x_, rs_y_, roff_x_, roff_y_;

    double rgb2evt_x(double u) const { return sr_x_ * u + off_x_; }
    double rgb2evt_y(double v) const { return sr_y_ * v + off_y_; }
    double evt2rgb_x(double u) const { return rs_x_ * u + roff_x_; }
    double evt2rgb_y(double v) const { return rs_y_ * v + roff_y_; }

    // --- Parameters ---
    std::string calib_path_;
    double evt_win_ms_;
    bool show_overlay_;
    bool show_trail_;
    bool gui_;
    double reseat_threshold_;

    // --- Tracker state ---
    enum Mode { IDLE, PAUSED, TRACKING };
    Mode mode_ = IDLE;

    cv::Ptr<cv::legacy::TrackerMOSSE> mosse_;
    cv::Rect2d bbox_, evt_bbox_;
    std::unique_ptr<EventBBoxPredictor> predictor_;
    BBoxTrail trail_;
    ReIdentifier reid_;
    bool reid_active_ = false;

    double event_density_ = 0, confidence_ = 0;

    // Latest data
    cv::Mat latest_frame_;
    std::vector<Event> latest_events_;
    double latest_evt_t0_ = 0, latest_evt_t1_ = 0;
    int evt_w_ = 0, evt_h_ = 0;
    bool predictor_initialized_ = false;

    // GUI mouse state
    struct SelectionState {
        bool selecting = false;
        bool selected  = false;
        cv::Point origin;
        cv::Rect roi;
    } sel_;

    // --- ROS 2 handles ---
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<event_tracker::msg::EventArray>::SharedPtr event_sub_;
    rclcpp::Publisher<event_tracker::msg::TrackingStatus>::SharedPtr status_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tsurf_pub_;

    static void onMouse(int event, int x, int y, int, void* ud) {
        auto* s = static_cast<SelectionState*>(ud);
        switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            s->selecting = true; s->selected = false;
            s->origin = {x, y}; s->roi = {x, y, 0, 0};
            break;
        case cv::EVENT_MOUSEMOVE:
            if (s->selecting)
                s->roi = cv::Rect(std::min(x, s->origin.x), std::min(y, s->origin.y),
                                   std::abs(x - s->origin.x), std::abs(y - s->origin.y));
            break;
        case cv::EVENT_LBUTTONUP:
            if (s->selecting) {
                s->roi = cv::Rect(std::min(x, s->origin.x), std::min(y, s->origin.y),
                                   std::abs(x - s->origin.x), std::abs(y - s->origin.y));
                s->selecting = false;
                if (s->roi.width > 5 && s->roi.height > 5) s->selected = true;
            }
            break;
        }
    }

    void eventCallback(const event_tracker::msg::EventArray::SharedPtr msg) {
        // Convert ROS events to internal Event format
        latest_events_.clear();
        latest_events_.reserve(msg->events.size());
        for (const auto& e : msg->events) {
            double t = e.stamp.sec + e.stamp.nanosec * 1e-9;
            latest_events_.emplace_back(t, e.x, e.y, e.polarity);
        }
        evt_w_ = msg->width;
        evt_h_ = msg->height;

        if (!latest_events_.empty()) {
            latest_evt_t0_ = latest_events_.front().t;
            latest_evt_t1_ = latest_events_.back().t;
        }

        if (!predictor_initialized_ && evt_w_ > 0 && evt_h_ > 0) {
            predictor_ = std::make_unique<EventBBoxPredictor>(evt_w_, evt_h_);
            predictor_initialized_ = true;
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS image to OpenCV
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }
        latest_frame_ = cv_ptr->image.clone();
        auto stamp = msg->header.stamp;

        if (latest_frame_.empty()) return;

        cv::Mat display = latest_frame_.clone();

        // --- Tracking logic (identical to standalone) ---
        if (mode_ == TRACKING && mosse_ && predictor_initialized_) {
            bool mosse_ok = mosse_->update(latest_frame_, bbox_);

            bool drifted = false;
            if (mosse_ok) {
                drifted = reid_.hasDrifted(bbox_, latest_frame_,
                    latest_frame_.cols, latest_frame_.rows);
            }

            // Event flow
            cv::Rect2d flow_bbox = (mosse_ok && !drifted) ? bbox_ : reid_.predictedBBox();
            cv::Rect2d evt_roi(
                rgb2evt_x(flow_bbox.x), rgb2evt_y(flow_bbox.y),
                flow_bbox.width * sr_x_, flow_bbox.height * sr_y_);

            cv::Point2d shift = predictor_->predict(
                latest_events_, latest_evt_t0_, latest_evt_t1_,
                evt_roi, event_density_, confidence_);
            shift.x *= rs_x_;
            shift.y *= rs_y_;

            if (mosse_ok && !drifted) {
                reid_active_ = false;

                cv::Rect2d smoothed = reid_.update(bbox_, shift, confidence_, latest_frame_);

                // Kalman-reseat
                double dx = (bbox_.x + bbox_.width/2) - (smoothed.x + smoothed.width/2);
                double dy = (bbox_.y + bbox_.height/2) - (smoothed.y + smoothed.height/2);
                double gap = std::sqrt(dx*dx + dy*dy);
                if (gap > reseat_threshold_) {
                    mosse_ = cv::legacy::TrackerMOSSE::create();
                    mosse_->init(latest_frame_, smoothed);
                    bbox_ = smoothed;
                }

                evt_bbox_ = bbox_;
                evt_bbox_.x += shift.x;
                evt_bbox_.y += shift.y;

                // Draw
                cv::rectangle(display, bbox_, {0,255,0}, 2);
                cv::rectangle(display, smoothed, {255,255,255}, 1);
                if (confidence_ > 0.15) {
                    cv::rectangle(display, evt_bbox_, {255,255,0}, 1);
                    cv::Point2d ctr(bbox_.x + bbox_.width/2, bbox_.y + bbox_.height/2);
                    double mag = std::sqrt(shift.x*shift.x + shift.y*shift.y);
                    double arrow_len = std::min(60.0, mag * 8.0);
                    if (mag > 0.3) {
                        double nx = shift.x / mag, ny = shift.y / mag;
                        cv::Point2d tip(ctr.x + nx*arrow_len, ctr.y + ny*arrow_len);
                        cv::arrowedLine(display, ctr, tip, {255,255,0}, 2, cv::LINE_AA, 0, 0.3);
                    }
                }

                trail_.push(bbox_);

                publishStatus(stamp, event_tracker::msg::TrackingStatus::MODE_TRACKING,
                              bbox_, smoothed, evt_bbox_, shift);
            } else {
                reid_active_ = true;
                cv::Rect2d recovered;

                if (reid_.recover(latest_frame_, recovered)) {
                    mosse_ = cv::legacy::TrackerMOSSE::create();
                    bbox_ = recovered;
                    evt_bbox_ = recovered;
                    mosse_->init(latest_frame_, bbox_);
                    reid_active_ = false;

                    cv::rectangle(display, recovered, {255,165,0}, 2);
                    cv::putText(display, "RE-ID OK", {10, display.rows-40},
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, {255,165,0}, 2);
                    trail_.push(recovered);

                    publishStatus(stamp, event_tracker::msg::TrackingStatus::MODE_RECOVERED,
                                  recovered, recovered, recovered, {0,0});
                } else {
                    cv::Rect2d pred = reid_.predictedBBox();
                    cv::rectangle(display, pred, {0,100,255}, 1);

                    char lostBuf[64];
                    snprintf(lostBuf, sizeof(lostBuf), "LOST (%d frames)%s",
                             reid_.lostFrames(), drifted ? " [drift]" : "");
                    cv::putText(display, lostBuf, {10, display.rows-40},
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,0,255}, 2);

                    if (reid_.gaveUp()) {
                        mosse_.release();
                        trail_.clear();
                        mode_ = IDLE;
                        RCLCPP_WARN(this->get_logger(),
                            "Re-ID gave up after %d frames", reid_.lostFrames());
                    }

                    publishStatus(stamp, event_tracker::msg::TrackingStatus::MODE_LOST,
                                  bbox_, pred, evt_bbox_, shift);
                }
            }
        }

        // Selection rectangle
        if (mode_ == PAUSED && (sel_.selecting || sel_.selected))
            cv::rectangle(display, sel_.roi, {255,100,0}, 2);

        if (show_trail_ && mode_ == TRACKING)
            trail_.draw(display, {0, 255, 0});

        // --- HUD ---
        std::string mstr;
        switch (mode_) {
            case IDLE:     mstr = "IDLE — SPACE to pause & select"; break;
            case PAUSED:   mstr = "PAUSED — draw box, then SPACE"; break;
            case TRACKING:
                mstr = reid_active_
                    ? "SEARCHING (Re-ID + Kalman)"
                    : "TRACKING (MOSSE + Events + Kalman)";
                break;
        }
        cv::putText(display, mstr, {10, 25}, cv::FONT_HERSHEY_SIMPLEX,
                    0.55, {0,255,255}, 1, cv::LINE_AA);

        // Publish debug image
        auto debug_msg = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", display).toImageMsg();
        debug_msg->header.stamp = stamp;
        debug_pub_->publish(*debug_msg);

        // Publish time surface
        if (predictor_initialized_) {
            double t_now = latest_evt_t1_;
            cv::Mat tsurf = predictor_->renderTimeSurface(t_now, 0.04);
            auto tsurf_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", tsurf).toImageMsg();
            tsurf_msg->header.stamp = stamp;
            tsurf_pub_->publish(*tsurf_msg);
        }

        // --- GUI (interactive window, same controls as standalone) ---
        if (gui_) {
            static bool gui_initialized = false;
            if (!gui_initialized) {
                cv::namedWindow("RGB + Tracking", cv::WINDOW_NORMAL);
                cv::resizeWindow("RGB + Tracking", 960, 600);
                cv::setMouseCallback("RGB + Tracking", onMouse, &sel_);
                gui_initialized = true;
            }

            cv::imshow("RGB + Tracking", display);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                rclcpp::shutdown();
            }
            if (key == ' ') {
                if (mode_ == IDLE) { mode_ = PAUSED; sel_ = {}; }
                else if (mode_ == PAUSED) {
                    if (sel_.selected) {
                        mosse_ = cv::legacy::TrackerMOSSE::create();
                        bbox_ = cv::Rect2d(sel_.roi);
                        evt_bbox_ = bbox_;
                        mosse_->init(latest_frame_, bbox_);
                        reid_.init(bbox_, latest_frame_);
                        reid_active_ = false;
                        trail_.clear();
                        mode_ = TRACKING;
                        RCLCPP_INFO(this->get_logger(),
                            "Tracking started: bbox=(%d,%d %dx%d)",
                            sel_.roi.x, sel_.roi.y, sel_.roi.width, sel_.roi.height);
                    } else { mode_ = IDLE; }
                } else { mode_ = PAUSED; sel_ = {}; }
            }
            if (key == 'r') {
                mosse_.release(); trail_.clear(); mode_ = IDLE;
                event_density_ = confidence_ = 0;
                reid_active_ = false;
                RCLCPP_INFO(this->get_logger(), "Tracker reset");
            }
            if (key == 'e') {
                show_overlay_ = !show_overlay_;
                RCLCPP_INFO(this->get_logger(), "Event overlay: %s",
                            show_overlay_ ? "ON" : "OFF");
            }
            if (key == 't') {
                show_trail_ = !show_trail_;
                if (!show_trail_) trail_.clear();
            }
        }
    }

    void publishStatus(const builtin_interfaces::msg::Time& stamp,
                       uint8_t mode,
                       const cv::Rect2d& bbox,
                       const cv::Rect2d& smoothed,
                       const cv::Rect2d& evt_bbox,
                       const cv::Point2d& shift) {
        auto msg = event_tracker::msg::TrackingStatus();
        msg.header.stamp = stamp;
        msg.header.frame_id = "rgb_camera";
        msg.mode = mode;

        msg.bbox_x = bbox.x;       msg.bbox_y = bbox.y;
        msg.bbox_w = bbox.width;    msg.bbox_h = bbox.height;

        msg.smoothed_x = smoothed.x;     msg.smoothed_y = smoothed.y;
        msg.smoothed_w = smoothed.width;  msg.smoothed_h = smoothed.height;

        msg.evt_bbox_x = evt_bbox.x;     msg.evt_bbox_y = evt_bbox.y;
        msg.evt_bbox_w = evt_bbox.width;  msg.evt_bbox_h = evt_bbox.height;

        msg.flow_dx = shift.x;
        msg.flow_dy = shift.y;

        auto vel = reid_.velocity();
        msg.vel_x = vel.x;
        msg.vel_y = vel.y;

        msg.event_confidence = confidence_;
        msg.event_density = event_density_;
        msg.appearance_score = reid_.lastAppearanceScore();
        msg.lost_frames = reid_.lostFrames();

        status_pub_->publish(msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}
