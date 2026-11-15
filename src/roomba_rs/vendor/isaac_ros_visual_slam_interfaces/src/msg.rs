#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus
/// This is a status message which gives insights about tracker status and execution time which can help in diagnostics and profiling.
/// The frame id in the header is used as the reference frame of both the transforms below.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisualSlamStatus {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    /// Pure visual odometry return code:
    /// 0 - Unknown state
    /// 1 - Success
    /// 2 - Failed
    pub vo_state: u8,

    /// The total time it takes to process the input frames to output messages in seconds.
    pub node_callback_execution_time: f64,

    /// Time it takes to get current pose out of cuVSLAM using pure visual slam tracking in seconds.
    pub track_execution_time: f64,

    /// Mean time to get poses out of cuVSLAM using pure visual slam tracking in seconds.
    pub track_execution_time_mean: f64,

    /// Max time to get poses out of cuVSLAM using pure visual slam tracking in seconds.
    pub track_execution_time_max: f64,

}



impl Default for VisualSlamStatus {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::VisualSlamStatus::default())
  }
}

impl rosidl_runtime_rs::Message for VisualSlamStatus {
  type RmwMsg = super::msg::rmw::VisualSlamStatus;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        vo_state: msg.vo_state,
        node_callback_execution_time: msg.node_callback_execution_time,
        track_execution_time: msg.track_execution_time,
        track_execution_time_mean: msg.track_execution_time_mean,
        track_execution_time_max: msg.track_execution_time_max,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      vo_state: msg.vo_state,
      node_callback_execution_time: msg.node_callback_execution_time,
      track_execution_time: msg.track_execution_time,
      track_execution_time_mean: msg.track_execution_time_mean,
      track_execution_time_max: msg.track_execution_time_max,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      vo_state: msg.vo_state,
      node_callback_execution_time: msg.node_callback_execution_time,
      track_execution_time: msg.track_execution_time,
      track_execution_time_mean: msg.track_execution_time_mean,
      track_execution_time_max: msg.track_execution_time_max,
    }
  }
}


