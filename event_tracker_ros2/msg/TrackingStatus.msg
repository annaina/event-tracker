# Tracking status published by the tracker node.

std_msgs/Header header

# Current mode
uint8 MODE_IDLE     = 0
uint8 MODE_TRACKING = 1
uint8 MODE_LOST     = 2
uint8 MODE_RECOVERED = 3
uint8 mode

# MOSSE bounding box (in RGB pixel coords)
float64 bbox_x
float64 bbox_y
float64 bbox_w
float64 bbox_h

# Kalman-smoothed bounding box
float64 smoothed_x
float64 smoothed_y
float64 smoothed_w
float64 smoothed_h

# Event-predicted bounding box
float64 evt_bbox_x
float64 evt_bbox_y
float64 evt_bbox_w
float64 evt_bbox_h

# Flow estimate from events (pixels/frame)
float64 flow_dx
float64 flow_dy

# Kalman velocity estimate (pixels/frame)
float64 vel_x
float64 vel_y

# Confidence metrics
float64 event_confidence     # 0-1, how much event data we have
float64 event_density        # events/sec inside bbox
float64 appearance_score     # 0-1, NCC against gallery

int32   lost_frames          # 0 if tracking, >0 if lost
