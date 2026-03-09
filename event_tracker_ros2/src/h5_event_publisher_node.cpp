// H5 Event & RGB Publisher Node
//
// Reads an M3ED HDF5 file and publishes events and RGB frames as ROS 2
// topics, simulating a live event camera + RGB camera setup. This replaces
// the direct H5 file reading that the standalone tracker did in its main().
//
// Published topics:
//   ~/events      (event_tracker/msg/EventArray)  — event batches
//   ~/image_raw   (sensor_msgs/msg/Image)         — RGB frames
//
// Parameters:
//   h5_path       — path to the M3ED .h5 file
//   side          — "left" or "right" event camera (default: "left")
//   start_sec     — start time offset (default: 0.0)
//   duration_sec  — how long to play, -1 = all (default: -1.0)
//   event_window_ms — ms of events per batch (default: 5.0)

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "event_tracker/event.h"
#include "event_tracker/rgb_reader.h"
#include "event_tracker/msg/event.hpp"
#include "event_tracker/msg/event_array.hpp"

class H5EventPublisher : public rclcpp::Node {
public:
    H5EventPublisher() : Node("h5_event_publisher") {
        // Declare parameters
        this->declare_parameter<std::string>("h5_path", "");
        this->declare_parameter<std::string>("side", "left");
        this->declare_parameter<double>("start_sec", 0.0);
        this->declare_parameter<double>("duration_sec", -1.0);
        this->declare_parameter<double>("event_window_ms", 5.0);

        h5_path_   = this->get_parameter("h5_path").as_string();
        side_      = this->get_parameter("side").as_string();
        start_sec_ = this->get_parameter("start_sec").as_double();
        duration_  = this->get_parameter("duration_sec").as_double();
        evt_win_ms_ = this->get_parameter("event_window_ms").as_double();

        if (h5_path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "h5_path parameter is required!");
            return;
        }

        // Open data sources
        if (!rgb_reader_.openH5(h5_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open H5 for RGB: %s",
                         h5_path_.c_str());
            return;
        }
        if (!event_reader_.open(h5_path_, side_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open H5 for events: %s",
                         h5_path_.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "Opened %s — RGB: %d frames @ %.1f fps (%dx%d), Events: %zu (%dx%d)",
            h5_path_.c_str(),
            rgb_reader_.numFrames(), rgb_reader_.fps(),
            rgb_reader_.width(), rgb_reader_.height(),
            event_reader_.totalEvents(),
            event_reader_.width(), event_reader_.height());

        // Publishers
        event_pub_ = this->create_publisher<event_tracker::msg::EventArray>(
            "~/events", 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "~/image_raw", 10);

        // Set up frame iteration
        frame_idx_ = (start_sec_ > 0)
            ? std::min((int)(start_sec_ * rgb_reader_.fps()),
                       rgb_reader_.numFrames() - 1)
            : 0;
        end_frame_ = rgb_reader_.numFrames();
        if (duration_ > 0)
            end_frame_ = std::min(end_frame_,
                frame_idx_ + (int)(duration_ * rgb_reader_.fps()));
        t0_ = rgb_reader_.startTime();

        // Timer drives the publishing at the RGB frame rate
        double period_ms = 1000.0 / rgb_reader_.fps();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds((int)period_ms),
            std::bind(&H5EventPublisher::publishFrame, this));

        RCLCPP_INFO(this->get_logger(),
            "Publishing frames %d–%d at %.1f fps, event window %.1f ms",
            frame_idx_, end_frame_, rgb_reader_.fps(), evt_win_ms_);
    }

private:
    void publishFrame() {
        if (frame_idx_ >= end_frame_) {
            RCLCPP_INFO(this->get_logger(), "Reached end of recording.");
            timer_->cancel();
            return;
        }

        double t_sec = t0_ + frame_idx_ / rgb_reader_.fps();
        auto stamp = this->now();

        // --- Publish RGB frame ---
        cv::Mat frame;
        if (rgb_reader_.getFrame(t_sec, frame) && !frame.empty()) {
            auto img_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            img_msg->header.stamp = stamp;
            img_msg->header.frame_id = "rgb_camera";
            image_pub_->publish(*img_msg);
        }

        // --- Publish event batch ---
        double evt_dt = evt_win_ms_ / 1000.0;
        double evt_t0 = std::max(0.0, t_sec - evt_dt);
        EventBatch batch;
        if (event_reader_.readBatch(evt_t0, evt_dt, batch) && !batch.empty()) {
            auto evt_msg = event_tracker::msg::EventArray();
            evt_msg.header.stamp = stamp;
            evt_msg.header.frame_id = "event_camera";
            evt_msg.width  = batch.width;
            evt_msg.height = batch.height;
            evt_msg.events.reserve(batch.events.size());

            for (const auto& e : batch.events) {
                event_tracker::msg::Event emsg;
                // Convert event time (seconds from t0) to ROS time
                int32_t sec = static_cast<int32_t>(e.t);
                uint32_t nsec = static_cast<uint32_t>((e.t - sec) * 1e9);
                emsg.stamp.sec = sec;
                emsg.stamp.nanosec = nsec;
                emsg.x = e.x;
                emsg.y = e.y;
                emsg.polarity = e.p;
                evt_msg.events.push_back(emsg);
            }
            event_pub_->publish(evt_msg);
        }

        frame_idx_++;
    }

    // Data sources
    std::string h5_path_;
    std::string side_;
    double start_sec_;
    double duration_;
    double evt_win_ms_;
    double t0_;

    RGBReader rgb_reader_;
    H5EventReader event_reader_;

    int frame_idx_;
    int end_frame_;

    // ROS 2
    rclcpp::Publisher<event_tracker::msg::EventArray>::SharedPtr event_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<H5EventPublisher>());
    rclcpp::shutdown();
    return 0;
}
