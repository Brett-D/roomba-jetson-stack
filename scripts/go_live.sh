#!/usr/bin/env bash
# Stops whatever is running (replay, mapping, navigation) and starts the normal live stack on
# the saved map ~/maps/vslam.db, plus navigation (web app "Go to"). Without navigation:
#   ~/go_live.sh --no-nav
pkill -INT -f "ros2 bag play" 2>/dev/null
~/stop_slam.sh > /dev/null
setsid nohup ~/run_vslam.sh > ~/vslam.log 2>&1 < /dev/null &
if [ "$1" != "--no-nav" ]; then
  # Nav2 needs the map and TF from the SLAM stack; give it time to come up first.
  setsid nohup bash -c 'sleep 30; exec ~/run_nav.sh' > ~/nav.log 2>&1 < /dev/null &
  NAV=" with navigation"
fi
echo "Live mode$NAV starting on ~/maps/vslam.db. Open http://$(hostname -I | awk '{print $1}'):8080 in about a minute."
