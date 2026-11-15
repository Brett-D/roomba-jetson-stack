#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__init(msg: *mut VisualSlamStatus) -> bool;
    fn isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<VisualSlamStatus>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<VisualSlamStatus>);
    fn isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<VisualSlamStatus>, out_seq: *mut rosidl_runtime_rs::Sequence<VisualSlamStatus>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This is a status message which gives insights about tracker status and execution time which can help in diagnostics and profiling.
/// The frame id in the header is used as the reference frame of both the transforms below.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VisualSlamStatus {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for VisualSlamStatus {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for VisualSlamStatus {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for VisualSlamStatus where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/msg/VisualSlamStatus";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__msg__VisualSlamStatus() }
  }
}


