// Offline Camera Calibration — ROS 2 Node
//
// Translates calibrate_main.cpp into a ROS 2 node. Offers the calibration
// as a ROS 2 service (~/calibrate) so other nodes can trigger it, and also
// supports running standalone via parameters (like the original tool).
//
// Service:
//   ~/calibrate  (event_tracker/srv/Calibrate) — run calibration
//
// Parameters (for standalone mode):
//   h5_path       — path to M3ED .h5 file
//   output_path   — where to save the YAML result (default: calibration.yaml)
//   side          — "left" or "right" (default: "left")
//   num_frames    — frames to sample (default: 20)
//   wide          — wider search for unknown cameras (default: false)
//   preview       — show visual overlay (default: false)
//   run_on_start  — run calibration immediately on startup (default: false)

#include <rclcpp/rclcpp.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iomanip>
#include <chrono>

#include "event_tracker/rgb_reader.h"
#include "event_tracker/event.h"
#include "event_tracker/auto_calibrate.h"
#include "event_tracker/srv/calibrate.hpp"

class CalibrateNode : public rclcpp::Node {
public:
    CalibrateNode() : Node("calibrate_node") {
        this->declare_parameter<std::string>("h5_path", "");
        this->declare_parameter<std::string>("output_path", "calibration.yaml");
        this->declare_parameter<std::string>("side", "left");
        this->declare_parameter<int>("num_frames", 20);
        this->declare_parameter<bool>("wide", false);
        this->declare_parameter<bool>("preview", false);
        this->declare_parameter<bool>("run_on_start", false);

        // Service
        service_ = this->create_service<event_tracker::srv::Calibrate>(
            "~/calibrate",
            std::bind(&CalibrateNode::handleCalibrate, this,
                      std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Calibrate node ready. Call ~/calibrate service.");

        // Optionally run immediately
        if (this->get_parameter("run_on_start").as_bool()) {
            std::string h5 = this->get_parameter("h5_path").as_string();
            if (!h5.empty()) {
                auto result = runCalibration(
                    h5,
                    this->get_parameter("side").as_string(),
                    this->get_parameter("num_frames").as_int(),
                    this->get_parameter("wide").as_bool(),
                    this->get_parameter("output_path").as_string(),
                    this->get_parameter("preview").as_bool());
                if (result.score > 0) {
                    RCLCPP_INFO(this->get_logger(),
                        "Calibration complete: sx=%.4f sy=%.4f ox=%.1f oy=%.1f (%d%% overlap)",
                        result.sx, result.sy, result.ox, result.oy,
                        (int)(result.score * 100));
                }
            }
        }
    }

private:
    rclcpp::Service<event_tracker::srv::Calibrate>::SharedPtr service_;

    void handleCalibrate(
        const std::shared_ptr<event_tracker::srv::Calibrate::Request> request,
        std::shared_ptr<event_tracker::srv::Calibrate::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Calibration requested for: %s",
                    request->h5_path.c_str());

        std::string side = request->side.empty() ? "left" : request->side;
        int nFrames = request->num_frames > 0 ? request->num_frames : 20;
        std::string outPath = request->output_path.empty()
            ? "calibration.yaml" : request->output_path;

        CalibResult result = runCalibration(
            request->h5_path, side, nFrames, request->wide, outPath, false);

        if (result.score > 0) {
            response->success = true;
            response->message = "Calibration complete";
            response->sx = result.sx;
            response->sy = result.sy;
            response->ox = result.ox;
            response->oy = result.oy;
            response->score = result.score;
        } else {
            response->success = false;
            response->message = "Calibration failed — no usable frame pairs";
        }
    }

    CalibResult runCalibration(const std::string& h5_path,
                               const std::string& side,
                               int nFrames, bool wide,
                               const std::string& outPath,
                               bool preview)
    {
        RCLCPP_INFO(this->get_logger(),
            "Running calibration: %s, side=%s, frames=%d, wide=%s",
            h5_path.c_str(), side.c_str(), nFrames, wide ? "yes" : "no");

        RGBReader reader;
        if (!reader.openH5(h5_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open H5 for RGB");
            return {0,0,0,0,0};
        }
        H5EventReader eventReader;
        if (!eventReader.open(h5_path, side)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open H5 for events");
            return {0,0,0,0,0};
        }

        int totalFrames = reader.numFrames();
        int n = std::min(nFrames, totalFrames);
        std::vector<cv::Mat> rgb_edges_vec, evt_edges_vec;

        RCLCPP_INFO(this->get_logger(), "Extracting %d edge pairs...", n);
        for (int i = 0; i < n; i++) {
            double frac = 0.1 + 0.8 * (double)i / std::max(1, n - 1);
            int fi = std::min(totalFrames - 1, (int)(totalFrames * frac));
            double t = reader.startTime() + fi / reader.fps();

            cv::Mat frm;
            if (!reader.getFrame(t, frm) || frm.empty()) continue;
            cv::Mat gray, edges;
            cv::cvtColor(frm, gray, cv::COLOR_BGR2GRAY);
            cv::Canny(gray, edges, 40, 120);

            EventBatch evts;
            double evt_win = 0.015;
            eventReader.readBatch(std::max(0.0, t - evt_win), evt_win * 2, evts);
            cv::Mat evt_edges = buildEventEdges(evts,
                eventReader.width(), eventReader.height());

            if (!edges.empty() && !evt_edges.empty()) {
                rgb_edges_vec.push_back(edges);
                evt_edges_vec.push_back(evt_edges);
            }
        }

        if (rgb_edges_vec.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No usable frame pairs found");
            return {0,0,0,0,0};
        }

        // Initial guess
        double init_sx, init_sy, init_ox, init_oy;
        if (wide) {
            init_sx = 0.85; init_sy = 0.85; init_ox = 0.0; init_oy = 0.0;
        } else {
            double fx_e = 1034.86, fy_e = 1033.48, cx_e = 629.70, cy_e = 357.60;
            double fx_r = 1268.56, fy_r = 1267.35, cx_r = 649.37, cy_r = 359.94;
            init_sx = fx_e / fx_r;
            init_sy = fy_e / fy_r;
            init_ox = cx_e - cx_r * init_sx;
            init_oy = cy_e - cy_r * init_sy;
        }

        CalibResult cal = autoCalibrate(
            rgb_edges_vec, evt_edges_vec,
            init_sx, init_sy, init_ox, init_oy, wide);

        // Save
        if (!outPath.empty()) {
            std::ofstream f(outPath);
            if (f.is_open()) {
                f << "# Camera calibration (event <-> RGB)\n"
                  << "# Generated by: event_tracker calibrate_node\n"
                  << "# Source file : " << h5_path << "\n"
                  << "#\n"
                  << "# Transform: u_evt = sx * u_rgb + ox\n"
                  << "#            v_evt = sy * v_rgb + oy\n"
                  << "#\n"
                  << std::fixed << std::setprecision(6)
                  << "sx: " << cal.sx << "\n"
                  << "sy: " << cal.sy << "\n"
                  << "ox: " << cal.ox << "\n"
                  << "oy: " << cal.oy << "\n"
                  << "score: " << cal.score << "\n";
                f.close();
                RCLCPP_INFO(this->get_logger(), "Saved calibration to: %s", outPath.c_str());
            }
        }

        if (preview) {
            // Show preview (blocking)
            cv::Mat frm;
            double t = reader.startTime() + (totalFrames / 2) / reader.fps();
            if (reader.getFrame(t, frm)) {
                cv::Mat gray, rgb_edges;
                cv::cvtColor(frm, gray, cv::COLOR_BGR2GRAY);
                cv::Canny(gray, rgb_edges, 40, 120);
                // Simple overlay
                cv::Mat vis = frm.clone();
                char buf[128];
                snprintf(buf, sizeof(buf),
                    "sx=%.4f sy=%.4f ox=%.1f oy=%.1f score=%d%%",
                    cal.sx, cal.sy, cal.ox, cal.oy, (int)(cal.score*100));
                cv::putText(vis, buf, {10,30}, cv::FONT_HERSHEY_SIMPLEX,
                            0.7, {255,255,255}, 2);
                cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
                cv::imshow("Calibration", vis);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
        }

        return cal;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CalibrateNode>());
    rclcpp::shutdown();
    return 0;
}
