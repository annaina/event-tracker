// Event-Enhanced MOSSE Tracker
//
// Fuses a standard MOSSE correlation-filter tracker (running on ~25fps RGB
// frames) with Prophesee event camera data to get motion information between
// frames. The event camera fires at microsecond resolution whenever a pixel
// changes brightness, so we can estimate optical flow inside the tracked
// bounding box even between RGB frames.
//
// The optical flow is computed from the "time-surface" -- a map of when each
// pixel last fired. By fitting a local plane to the time-surface around each
// event, we get a per-event flow estimate. Averaging these over the bbox and
// smoothing across frames gives a stable motion arrow.
//
// Two windows:
//   Left  -- RGB frame with MOSSE bbox, event-predicted bbox, flow arrow
//   Right -- Event time-surface (green = ON events, red = OFF events)

#include <iostream>
#include <string>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include "rgb_reader.h"
#include "event.h"
#include "reidentifier.h"
#include "auto_calibrate.h"

// ---------------------------------------------------------------------------
//  Command-line arguments
// ---------------------------------------------------------------------------

struct Args {
    std::string h5path;
    std::string side      = "left";
    std::string calibPath;            // path to calibration.yaml
    double      startSec  = 0.0;
    double      duration  = -1.0;   // negative means play everything
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.h5> [--calib calibration.yaml]"
                  << " [--side left|right] [--start S] [--duration S]\n";
        std::exit(1);
    }
    a.h5path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--side" && i + 1 < argc) a.side = argv[++i];
        else if (arg == "--start" && i + 1 < argc) a.startSec = std::stod(argv[++i]);
        else if (arg == "--duration" && i + 1 < argc) a.duration = std::stod(argv[++i]);
        else if (arg == "--calib" && i + 1 < argc) a.calibPath = argv[++i];
    }
    return a;
}

// ---------------------------------------------------------------------------
//  Mouse callback for drawing a bounding box
// ---------------------------------------------------------------------------

struct SelectionState {
    bool      selecting = false;
    bool      selected  = false;
    cv::Point origin;
    cv::Rect  roi;
};

static void onMouse(int event, int x, int y, int, void* ud) {
    auto* s = static_cast<SelectionState*>(ud);
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        s->selecting = true;
        s->selected  = false;
        s->origin    = {x, y};
        s->roi       = {x, y, 0, 0};
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

// ---------------------------------------------------------------------------
//  EventBBoxPredictor
//
//  This is where the event-camera magic happens. Given a bounding box and
//  a batch of events, we estimate the optical flow inside the box using
//  time-surface gradients.
//
//  The "time-surface" is just a map storing the most recent timestamp at
//  each pixel. When an edge moves across the sensor, nearby pixels fire
//  in sequence -- so the spatial gradient of the time-surface tells us
//  which direction and how fast things are moving.
//
//  For each event inside the bbox, we fit a plane T(x,y) = ax + by + c
//  to its 5x5 neighborhood on the time-surface. The gradient (a, b)
//  gives us the normal flow: v = -grad(T) / |grad(T)|^2.
//
//  We average all the per-event flows and smooth with an exponential
//  moving average so the arrow doesn't jump around frame-to-frame.
// ---------------------------------------------------------------------------

class EventBBoxPredictor {
public:
    EventBBoxPredictor(int img_w, int img_h)
        : img_w_(img_w), img_h_(img_h),
          time_surface_(img_h, img_w, CV_64FC1, cv::Scalar(0)),
          smooth_vx_(0), smooth_vy_(0) {}

    // Returns pixel displacement (dx, dy) in event-camera coordinates.
    // Also fills in event_density (events/sec) and confidence (0-1).
    cv::Point2d predict(const EventBatch& events, const cv::Rect2d& bbox,
                        double& event_density, double& confidence) {
        if (events.empty() || bbox.width <= 0) {
            event_density = 0;
            confidence = 0;
            smooth_vx_ *= 0.5;
            smooth_vy_ *= 0.5;
            return {smooth_vx_, smooth_vy_};
        }

        // Clamp bbox to image bounds (with 1px margin for gradient neighborhood)
        int bx0 = std::max(1, (int)bbox.x);
        int by0 = std::max(1, (int)bbox.y);
        int bx1 = std::min(img_w_ - 2, (int)(bbox.x + bbox.width));
        int by1 = std::min(img_h_ - 2, (int)(bbox.y + bbox.height));

        double t_min = events.events.front().t;
        double t_max = events.events.back().t;
        double dt = t_max - t_min;
        int total_in_bbox = 0;

        double sum_vx = 0, sum_vy = 0;
        int n_flow = 0;

        constexpr int R = 2;           // neighborhood radius for plane fit
        double recency = 0.030;        // ignore stale surface values (>30ms old)

        for (const auto& e : events.events) {
            // Always update the time-surface, even for events outside the bbox
            if (e.x >= 0 && e.x < img_w_ && e.y >= 0 && e.y < img_h_)
                time_surface_.at<double>(e.y, e.x) = e.t;

            // Only compute flow for events inside the tracked region
            if (e.x < bx0 || e.x > bx1 || e.y < by0 || e.y > by1)
                continue;
            total_in_bbox++;

            // Skip events too close to the image border for the neighborhood
            if (e.x < R || e.x >= img_w_ - R || e.y < R || e.y >= img_h_ - R)
                continue;

            // Fit a plane T = a*dx + b*dy + c to the local time-surface.
            // This is just a least-squares solve of a tiny 2-parameter system.
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

            if (count < 6) continue;    // not enough neighbors

            // Solve the 2x2 normal equations for the gradient (a, b)
            double n = count;
            double Sxx = sum_xx - sum_x*sum_x/n;
            double Syy = sum_yy - sum_y*sum_y/n;
            double Sxy = sum_xy - sum_x*sum_y/n;
            double Sxt = sum_xt - sum_x*sum_t/n;
            double Syt = sum_yt - sum_y*sum_t/n;

            double det = Sxx*Syy - Sxy*Sxy;
            if (std::abs(det) < 1e-20) continue;

            double a = (Syy*Sxt - Sxy*Syt) / det;   // dT/dx
            double b = (Sxx*Syt - Sxy*Sxt) / det;   // dT/dy

            double grad_sq = a*a + b*b;
            if (grad_sq < 1e-16) continue;

            // Normal optical flow: velocity = -gradient / |gradient|^2
            double vx = -a / grad_sq;
            double vy = -b / grad_sq;

            // Toss obvious outliers (anything faster than 2000 px/s is noise)
            double speed = std::sqrt(vx*vx + vy*vy);
            if (speed > 2000.0) continue;

            sum_vx += vx;
            sum_vy += vy;
            n_flow++;
        }

        event_density = (dt > 0) ? total_in_bbox / dt : 0;

        // If we didn't get enough good flow estimates, just decay the old value
        if (n_flow < 10) {
            double area = bbox.width * bbox.height;
            confidence = std::min(1.0, total_in_bbox / (area * 0.5));
            smooth_vx_ *= 0.7;
            smooth_vy_ *= 0.7;
            return {smooth_vx_ * dt, smooth_vy_ * dt};
        }

        // Average the per-event flow vectors (units: px/s)
        double avg_vx = sum_vx / n_flow;
        double avg_vy = sum_vy / n_flow;

        // Smooth across frames so the arrow doesn't jitter
        constexpr double alpha = 0.3;
        smooth_vx_ = alpha * avg_vx + (1.0 - alpha) * smooth_vx_;
        smooth_vy_ = alpha * avg_vy + (1.0 - alpha) * smooth_vy_;

        double area = bbox.width * bbox.height;
        double density_ratio = total_in_bbox / area;
        confidence = std::min(1.0, density_ratio / 0.5);

        // Convert velocity (px/s) to displacement (px) over this time window
        return {smooth_vx_ * dt, smooth_vy_ * dt};
    }

    // Render the time-surface as a color image.
    // Recent ON events show green, recent OFF events show red,
    // everything else is dark. Events fade over decay_sec.
    cv::Mat renderTimeSurface(double t_now, double decay_sec = 0.05) const {
        cv::Mat vis(img_h_, img_w_, CV_8UC3, cv::Scalar(15, 15, 15));
        for (int y = 0; y < img_h_; ++y) {
            for (int x = 0; x < img_w_; ++x) {
                double ts = time_surface_.at<double>(y, x);
                if (ts <= 0) continue;
                double age = t_now - ts;
                if (age < decay_sec) {
                    int v = (int)(255.0 * (1.0 - age / decay_sec));
                    int pol = polarity_surface_.empty() ? 1 :
                              (int)polarity_surface_.at<uchar>(y, x);
                    if (pol > 0)
                        vis.at<cv::Vec3b>(y, x) = {0, (uchar)v, 0};
                    else
                        vis.at<cv::Vec3b>(y, x) = {0, 0, (uchar)v};
                }
            }
        }
        return vis;
    }

    // Keep track of polarity (ON vs OFF) so the time-surface viz looks nice
    void updatePolaritySurface(const EventBatch& events) {
        if (polarity_surface_.empty())
            polarity_surface_ = cv::Mat(img_h_, img_w_, CV_8UC1, cv::Scalar(1));
        for (const auto& e : events.events) {
            if (e.x >= 0 && e.x < img_w_ && e.y >= 0 && e.y < img_h_)
                polarity_surface_.at<uchar>(e.y, e.x) = (e.p > 0) ? 1 : 0;
        }
    }

    // Build a heatmap of recent event activity (for overlaying on RGB)
    cv::Mat renderActivityMap(const EventBatch& events, double decay_sec = 0.033) const {
        cv::Mat heat(img_h_, img_w_, CV_32F, cv::Scalar(0));
        if (events.empty()) return heat;
        double t_now = events.events.back().t;
        for (const auto& e : events.events) {
            double age = t_now - e.t;
            if (age < decay_sec && e.x < img_w_ && e.y < img_h_)
                heat.at<float>(e.y, e.x) += (float)(1.0 - age / decay_sec);
        }
        double maxVal;
        cv::minMaxLoc(heat, nullptr, &maxVal);
        if (maxVal > 0) heat /= maxVal;
        cv::GaussianBlur(heat, heat, cv::Size(7, 7), 2.0);
        return heat;
    }

private:
    int img_w_, img_h_;
    cv::Mat time_surface_;
    cv::Mat polarity_surface_;
    double smooth_vx_, smooth_vy_;
};

// ---------------------------------------------------------------------------
//  Trail -- draws the recent bbox positions as a fading dotted path
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
//  HUD overlay -- status text, event stats, frame counter
// ---------------------------------------------------------------------------

static void drawHUD(cv::Mat& img, const std::string& mode_str,
                    double event_density, double confidence,
                    int frameIdx, int totalFrames, double evtWindowMs,
                    bool show_overlay) {
    int y = 25;
    auto put = [&](const std::string& text, cv::Scalar col) {
        cv::putText(img, text, {10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 1, cv::LINE_AA);
        y += 22;
    };
    put(mode_str, {0, 255, 255});
    if (event_density > 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Evt density: %.0f ev/s  Conf: %.0f%%",
                 event_density, confidence * 100);
        put(buf, confidence > 0.5 ? cv::Scalar(0,255,0) : cv::Scalar(0,128,255));
    }
    { char buf[64]; snprintf(buf, sizeof(buf), "Evt window: %.0f ms", evtWindowMs);
      put(buf, {180,180,180}); }
    if (show_overlay) put("[E] Event overlay ON", {0, 200, 200});
    { char buf[64]; snprintf(buf, sizeof(buf), "%d / %d", frameIdx, totalFrames);
      cv::putText(img, buf, {img.cols-150, img.rows-12},
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, {180,180,180}, 1, cv::LINE_AA); }
}

// ---------------------------------------------------------------------------
//  Blend an event heatmap on top of an RGB frame
// ---------------------------------------------------------------------------

static void overlayHeatmap(cv::Mat& rgb, const cv::Mat& heat32f, double alpha = 0.4) {
    if (heat32f.empty()) return;
    cv::Mat h;
    if (heat32f.size() != rgb.size()) cv::resize(heat32f, h, rgb.size());
    else h = heat32f;
    cv::Mat h8u; h.convertTo(h8u, CV_8U, 255.0);
    cv::Mat hColor; cv::applyColorMap(h8u, hColor, cv::COLORMAP_HOT);
    cv::Mat mask; cv::threshold(h8u, mask, 10, 255, cv::THRESH_BINARY);
    for (int y = 0; y < rgb.rows; ++y)
        for (int x = 0; x < rgb.cols; ++x)
            if (mask.at<uchar>(y,x)) {
                auto& p = rgb.at<cv::Vec3b>(y,x);
                auto& c = hColor.at<cv::Vec3b>(y,x);
                p[0] = cv::saturate_cast<uchar>(p[0]*(1-alpha) + c[0]*alpha);
                p[1] = cv::saturate_cast<uchar>(p[1]*(1-alpha) + c[1]*alpha);
                p[2] = cv::saturate_cast<uchar>(p[2]*(1-alpha) + c[2]*alpha);
            }
}

// ===========================================================================
//  main
// ===========================================================================

int main(int argc, char** argv)
{
    Args args = parseArgs(argc, argv);

    // Open the H5 file for both RGB frames and events
    RGBReader reader;
    if (!reader.openH5(args.h5path)) {
        std::cerr << "Failed to open H5 for RGB: " << args.h5path << "\n";
        return 1;
    }
    H5EventReader eventReader;
    if (!eventReader.open(args.h5path, args.side)) {
        std::cerr << "Failed to open H5 for events: " << args.h5path << "\n";
        return 1;
    }

    std::cout << "\n======================================================\n"
              << "  Event-Enhanced MOSSE Tracker\n"
              << "  RGB : " << reader.numFrames() << " frames @ "
              << reader.fps() << " fps  (" << reader.width() << "x"
              << reader.height() << ")\n"
              << "  Evt : " << eventReader.totalEvents() << " events  "
              << eventReader.width() << "x" << eventReader.height()
              << "  (" << args.side << ")\n"
              << "======================================================\n"
              << "  SPACE = pause / select bbox / start tracking\n"
              << "  R = reset   E = event overlay   T = trail\n"
              << "  +/- = event window   Q = quit\n"
              << "  Kalman filter + template re-ID active\n"
              << "======================================================\n\n";

    // -- Load camera calibration --
    //
    // The calibration maps event camera coords to RGB camera coords:
    //   u_evt = sx * u_rgb + ox
    //   v_evt = sy * v_rgb + oy
    //
    // Run the `calibrate` tool on a recording to generate the .yaml file:
    //   ./calibrate <data.h5> -o calibration.yaml
    // Then pass it here with --calib.

    double sr_x, sr_y, off_x, off_y;

    if (!args.calibPath.empty()) {
        CalibResult cal;
        if (!loadCalibration(args.calibPath, cal)) {
            std::cerr << "Error: cannot load calibration file: " << args.calibPath << "\n";
            return 1;
        }
        sr_x  = cal.sx;
        sr_y  = cal.sy;
        off_x = cal.ox;
        off_y = cal.oy;
        std::cout << "  Loaded calibration from: " << args.calibPath
                  << "  (score=" << (int)(cal.score*100) << "%)\n";
    } else {
        // Fallback: hardcoded M3ED falcon intrinsics (only valid for that rig)
        double fx_e = 1034.86, fy_e = 1033.48, cx_e = 629.70, cy_e = 357.60;
        double fx_r = 1268.56, fy_r = 1267.35, cx_r = 649.37, cy_r = 359.94;
        sr_x  = fx_e / fx_r;
        sr_y  = fy_e / fy_r;
        off_x = cx_e - cx_r * sr_x;
        off_y = cy_e - cy_r * sr_y;
        std::cout << "  WARNING: No calibration file loaded (use --calib).\n"
                  << "  Falling back to hardcoded M3ED intrinsics.\n"
                  << "  Run:  ./calibrate <data.h5> -o calibration.yaml\n";
    }

    double rs_x   = 1.0 / sr_x;
    double rs_y   = 1.0 / sr_y;
    double roff_x = -off_x / sr_x;
    double roff_y = -off_y / sr_y;

    auto rgb2evt_x = [&](double u) { return sr_x * u + off_x; };
    auto rgb2evt_y = [&](double v) { return sr_y * v + off_y; };
    auto evt2rgb_x = [&](double u) { return rs_x * u + roff_x; };
    auto evt2rgb_y = [&](double v) { return rs_y * v + roff_y; };

    std::cout << "  Calibration: RGB->Evt scale=(" << sr_x << "," << sr_y
              << ") offset=(" << off_x << "," << off_y << ")\n\n";

    // -- Create display windows --
    const std::string winRGB = "RGB + Tracking";
    const std::string winEvt = "Event Time-Surface";
    cv::namedWindow(winRGB, cv::WINDOW_NORMAL);
    cv::namedWindow(winEvt, cv::WINDOW_NORMAL);
    cv::resizeWindow(winRGB, 960, 600);
    cv::resizeWindow(winEvt, 960, 600);
    cv::moveWindow(winRGB, 50, 50);
    cv::moveWindow(winEvt, 1020, 50);

    SelectionState sel;
    cv::setMouseCallback(winRGB, onMouse, &sel);

    // -- Tracker state --
    enum Mode { PLAYING, PAUSED, TRACKING };
    Mode mode = PLAYING;

    cv::Ptr<cv::legacy::TrackerMOSSE> tracker;
    cv::Rect2d bbox, evt_bbox;
    EventBBoxPredictor predictor(eventReader.width(), eventReader.height());
    BBoxTrail trail;
    ReIdentifier reid;
    bool reid_active = false;   // true while we're in "lost" recovery mode

    double event_density = 0, confidence = 0;
    double evtWindowMs = 5.0;
    bool show_evt_overlay = true, show_trail = true;

    int frameIdx = 0;
    const int totalFrames = reader.numFrames();
    const int delayMs = std::max(1, (int)(1000.0 / reader.fps()));
    const double t0 = reader.startTime();

    if (args.startSec > 0)
        frameIdx = std::min((int)(args.startSec * reader.fps()), totalFrames - 1);
    int endFrame = totalFrames;
    if (args.duration > 0)
        endFrame = std::min(totalFrames, frameIdx + (int)(args.duration * reader.fps()));

    cv::Mat frame, display, evtVis;

    // -- Main loop: one iteration per RGB frame --
    while (frameIdx < endFrame) {
        double t_sec = t0 + frameIdx / reader.fps();

        if (!reader.getFrame(t_sec, frame) || frame.empty()) { frameIdx++; continue; }
        display = frame.clone();

        // Grab events for the current time window
        double evt_dt = evtWindowMs / 1000.0;
        double evt_t0 = std::max(0.0, t_sec - evt_dt);
        EventBatch events;
        eventReader.readBatch(evt_t0, evt_dt, events);
        predictor.updatePolaritySurface(events);

        // -- Run tracker --
        if (mode == TRACKING) {
            bool mosse_ok = tracker->update(frame, bbox);

            // Even if MOSSE says "ok", check for drift. MOSSE almost never
            // reports failure — it just silently drifts onto background.
            bool drifted = false;
            if (mosse_ok) {
                drifted = reid.hasDrifted(bbox, frame, frame.cols, frame.rows);
            }

            // Get event flow from the current bbox position
            cv::Rect2d flow_bbox = (mosse_ok && !drifted) ? bbox : reid.predictedBBox();
            cv::Rect2d evt_roi(
                rgb2evt_x(flow_bbox.x), rgb2evt_y(flow_bbox.y),
                flow_bbox.width * sr_x, flow_bbox.height * sr_y);
            cv::Point2d shift = predictor.predict(events, evt_roi,
                                                  event_density, confidence);
            shift.x *= rs_x;
            shift.y *= rs_y;

            if (mosse_ok && !drifted) {
                reid_active = false;

                // Feed MOSSE + events into the Kalman filter / re-ID module
                cv::Rect2d smoothed = reid.update(bbox, shift, confidence, frame);

                // The Kalman-smoothed bbox fuses MOSSE + events. If MOSSE
                // is drifting away from it, re-seat MOSSE on the Kalman
                // position so it doesn't wander off into background.
                double dx = (bbox.x + bbox.width/2) - (smoothed.x + smoothed.width/2);
                double dy = (bbox.y + bbox.height/2) - (smoothed.y + smoothed.height/2);
                double gap = std::sqrt(dx*dx + dy*dy);
                if (gap > 15.0) {
                    // MOSSE has drifted >15px from the Kalman estimate.
                    // Re-seat it. This is gentle — it only fires when
                    // there's real disagreement, not every frame.
                    tracker = cv::legacy::TrackerMOSSE::create();
                    tracker->init(frame, smoothed);
                    bbox = smoothed;
                }

                evt_bbox = bbox;
                evt_bbox.x += shift.x;
                evt_bbox.y += shift.y;

                // Draw MOSSE bbox (green) and Kalman-smoothed bbox (white)
                cv::rectangle(display, bbox, {0,255,0}, 2);
                cv::rectangle(display, smoothed, {255,255,255}, 1);

                // Draw event-predicted bbox and flow arrow (cyan/yellow)
                if (confidence > 0.15) {
                    cv::rectangle(display, evt_bbox, {255,255,0}, 1);
                    cv::Point2d ctr(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
                    double mag = std::sqrt(shift.x*shift.x + shift.y*shift.y);
                    double arrow_len = std::min(60.0, mag * 8.0);
                    if (mag > 0.3) {
                        double nx = shift.x / mag, ny = shift.y / mag;
                        cv::Point2d tip(ctr.x + nx*arrow_len, ctr.y + ny*arrow_len);
                        cv::arrowedLine(display, ctr, tip, {255,255,0}, 2, cv::LINE_AA, 0, 0.3);
                    }
                }

                // Appearance + confidence bars (top-right corner)
                int bw=100, bh=8, bx=display.cols-bw-10, by=10;
                cv::rectangle(display, {bx,by,bw,bh}, {80,80,80}, -1);
                int fill = (int)(confidence * bw);
                cv::rectangle(display, {bx,by,fill,bh},
                    confidence > 0.5 ? cv::Scalar(0,255,0) : cv::Scalar(0,140,255), -1);
                cv::putText(display, "evt conf", {bx, by+bh+14},
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, {180,180,180}, 1);

                // Appearance score bar
                by += 30;
                double app = reid.lastAppearanceScore();
                cv::rectangle(display, {bx,by,bw,bh}, {80,80,80}, -1);
                int afill = (int)(std::max(0.0, app) * bw);
                cv::rectangle(display, {bx,by,afill,bh},
                    app > 0.4 ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), -1);
                cv::putText(display, "appear", {bx, by+bh+14},
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, {180,180,180}, 1);

                trail.push(bbox);
            } else {
                // MOSSE lost or drifted — try to re-identify
                reid_active = true;
                cv::Rect2d recovered;

                if (reid.recover(frame, recovered)) {
                    // Re-ID found it! Re-init MOSSE at the recovered location.
                    tracker = cv::legacy::TrackerMOSSE::create();
                    bbox = recovered;
                    evt_bbox = recovered;
                    tracker->init(frame, bbox);
                    reid_active = false;

                    cv::rectangle(display, recovered, {255,165,0}, 2);
                    cv::putText(display, "RE-ID OK", {10, display.rows-40},
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, {255,165,0}, 2);
                    trail.push(recovered);
                } else {
                    // Still lost — show the Kalman prediction so the user
                    // can see where we think the target went
                    cv::Rect2d pred = reid.predictedBBox();
                    cv::rectangle(display, pred, {0,100,255}, 1);

                    char lostBuf[64];
                    snprintf(lostBuf, sizeof(lostBuf), "LOST (%d frames)%s",
                             reid.lostFrames(),
                             drifted ? " [drift]" : "");
                    cv::putText(display, lostBuf, {10, display.rows-40},
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,0,255}, 2);

                    // Draw the velocity arrow on the prediction
                    cv::Point2d vel = reid.velocity();
                    double vmag = std::sqrt(vel.x*vel.x + vel.y*vel.y);
                    if (vmag > 0.5) {
                        cv::Point2d pc(pred.x + pred.width/2, pred.y + pred.height/2);
                        cv::Point2d vt(pc.x + vel.x * 5.0, pc.y + vel.y * 5.0);
                        cv::arrowedLine(display, pc, vt, {0,100,255}, 2, cv::LINE_AA, 0, 0.3);
                    }

                    // If we've been lost too long, give up and reset
                    if (reid.gaveUp()) {
                        tracker.release();
                        trail.clear();
                        mode = PLAYING;
                        event_density = confidence = 0;
                        std::cout << "[Re-ID] Gave up after " << reid.lostFrames()
                                  << " frames.\n";
                    }
                }
            }
        }

        // Show the selection rectangle while the user is drawing
        if (mode == PAUSED && (sel.selecting || sel.selected))
            cv::rectangle(display, sel.roi, {255,100,0}, 2);

        // -- Event heatmap overlay on RGB --
        if (show_evt_overlay && !events.empty()) {
            cv::Mat heat = predictor.renderActivityMap(events, evt_dt);
            // Warp the heatmap from event coords to RGB coords
            cv::Mat heat_rgb(reader.height(), reader.width(), CV_32F, cv::Scalar(0));
            for (int ey = 0; ey < heat.rows; ++ey) {
                int ry = (int)std::round(evt2rgb_y(ey));
                if (ry < 0 || ry >= heat_rgb.rows) continue;
                for (int ex = 0; ex < heat.cols; ++ex) {
                    float v = heat.at<float>(ey, ex);
                    if (v < 0.01f) continue;
                    int rx = (int)std::round(evt2rgb_x(ex));
                    if (rx >= 0 && rx < heat_rgb.cols)
                        heat_rgb.at<float>(ry, rx) = std::max(heat_rgb.at<float>(ry, rx), v);
                }
            }
            cv::GaussianBlur(heat_rgb, heat_rgb, cv::Size(3, 3), 1.0);
            overlayHeatmap(display, heat_rgb, 0.35);
        }

        if (show_trail && mode == TRACKING)
            trail.draw(display, {0, 255, 0});

        // -- HUD --
        std::string mstr;
        switch (mode) {
            case PLAYING:  mstr = "PLAYING"; break;
            case PAUSED:   mstr = "PAUSED - draw box, then SPACE"; break;
            case TRACKING:
                mstr = reid_active
                    ? "SEARCHING (Re-ID + Kalman)"
                    : "TRACKING (MOSSE + Events + Kalman)";
                break;
        }
        drawHUD(display, mstr, event_density, confidence,
                frameIdx, totalFrames, evtWindowMs, show_evt_overlay);
        cv::imshow(winRGB, display);

        // -- Event time-surface window --
        evtVis = predictor.renderTimeSurface(t_sec, 0.04);
        if (mode == TRACKING) {
            cv::Rect2d er(rgb2evt_x(bbox.x), rgb2evt_y(bbox.y),
                          bbox.width * sr_x, bbox.height * sr_y);
            cv::rectangle(evtVis, er, {255,255,255}, 1);
        }
        { char buf[64];
          snprintf(buf, sizeof(buf), "%zu events", events.size());
          cv::putText(evtVis, buf, {10,25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200,200,200}, 1);
          snprintf(buf, sizeof(buf), "t=%.3f s", t_sec);
          cv::putText(evtVis, buf, {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200,200,200}, 1);
        }
        cv::imshow(winEvt, evtVis);

        // -- Keyboard controls --
        int key = cv::waitKey((mode==PLAYING||mode==TRACKING) ? delayMs : 30) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == ' ') {
            if (mode == PLAYING) { mode = PAUSED; sel = {}; }
            else if (mode == PAUSED) {
                if (sel.selected) {
                    tracker = cv::legacy::TrackerMOSSE::create();
                    bbox = cv::Rect2d(sel.roi);
                    evt_bbox = bbox;
                    tracker->init(frame, bbox);
                    reid.init(bbox, frame);
                    reid_active = false;
                    trail.clear();
                    mode = TRACKING;
                    std::cout << "Tracking started at frame " << frameIdx
                              << "  bbox=(" << sel.roi.x << "," << sel.roi.y
                              << " " << sel.roi.width << "x" << sel.roi.height << ")\n";
                } else { mode = PLAYING; }
            } else { mode = PAUSED; sel = {}; }
        }
        if (key == 'r') {
            tracker.release(); trail.clear(); mode = PLAYING;
            event_density = confidence = 0;
            reid_active = false;
            std::cout << "Tracker reset\n";
        }
        if (key == 'e') {
            show_evt_overlay = !show_evt_overlay;
            std::cout << "Event overlay: " << (show_evt_overlay?"ON":"OFF") << "\n";
        }
        if (key == 't') {
            show_trail = !show_trail;
            if (!show_trail) trail.clear();
            std::cout << "Trail: " << (show_trail?"ON":"OFF") << "\n";
        }
        if (key == '+' || key == '=') {
            evtWindowMs = std::min(100.0, evtWindowMs + 1.0);
            std::cout << "Event window: " << evtWindowMs << " ms\n";
        }
        if (key == '-') {
            evtWindowMs = std::max(1.0, evtWindowMs - 1.0);
            std::cout << "Event window: " << evtWindowMs << " ms\n";
        }
        if (mode != PAUSED) frameIdx++;
    }

    cv::destroyAllWindows();
    std::cout << "\nDone.\n";
    return 0;
}
