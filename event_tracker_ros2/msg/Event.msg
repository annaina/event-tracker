# A single event from a Prophesee event camera.
# Each event says "pixel (x,y) changed brightness at time t with polarity p".

builtin_interfaces/Time stamp   # ROS timestamp
uint16 x                        # pixel column
uint16 y                        # pixel row
int8   polarity                 # 1=ON (brighter), 0=OFF (darker)
