#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath_Request() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Request__init(msg: *mut FilePath_Request) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<FilePath_Request>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<FilePath_Request>);
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<FilePath_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<FilePath_Request>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__FilePath_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FilePath_Request {
    /// Request
    pub file_path: rosidl_runtime_rs::String,

}



impl Default for FilePath_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__FilePath_Request__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__FilePath_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for FilePath_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for FilePath_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for FilePath_Request where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/FilePath_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath_Request() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath_Response() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Response__init(msg: *mut FilePath_Response) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<FilePath_Response>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<FilePath_Response>);
    fn isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<FilePath_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<FilePath_Response>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__FilePath_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FilePath_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for FilePath_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__FilePath_Response__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__FilePath_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for FilePath_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__FilePath_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for FilePath_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for FilePath_Response where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/FilePath_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath_Response() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__init(msg: *mut GetAllPoses_Request) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Request>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Request>);
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<GetAllPoses_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Request>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GetAllPoses_Request {
    /// Max number of poses to query
    pub max_count: i32,

}



impl Default for GetAllPoses_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for GetAllPoses_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for GetAllPoses_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for GetAllPoses_Request where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/GetAllPoses_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__init(msg: *mut GetAllPoses_Response) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Response>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Response>);
    fn isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<GetAllPoses_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<GetAllPoses_Response>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GetAllPoses_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

    /// List of optimized poses
    pub poses: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::PoseStamped>,

}



impl Default for GetAllPoses_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for GetAllPoses_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for GetAllPoses_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for GetAllPoses_Response where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/GetAllPoses_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__init(msg: *mut LocalizeInMap_Request) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Request>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Request>);
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LocalizeInMap_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Request>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LocalizeInMap_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub map_folder_path: rosidl_runtime_rs::String,

    /// Pose hint used to localize in the map. Pose hint corresponds to map_T_robot.
    pub pose_hint: geometry_msgs::msg::rmw::Pose,

}



impl Default for LocalizeInMap_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LocalizeInMap_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LocalizeInMap_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LocalizeInMap_Request where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/LocalizeInMap_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__init(msg: *mut LocalizeInMap_Response) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Response>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Response>);
    fn isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LocalizeInMap_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<LocalizeInMap_Response>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LocalizeInMap_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for LocalizeInMap_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LocalizeInMap_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LocalizeInMap_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LocalizeInMap_Response where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/LocalizeInMap_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset_Request() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Request__init(msg: *mut Reset_Request) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Reset_Request>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Reset_Request>);
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Reset_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<Reset_Request>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__Reset_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Reset_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub structure_needs_at_least_one_member: u8,

}



impl Default for Reset_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__Reset_Request__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__Reset_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Reset_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Reset_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Reset_Request where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/Reset_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset_Request() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset_Response() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Response__init(msg: *mut Reset_Response) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Reset_Response>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Reset_Response>);
    fn isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Reset_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<Reset_Response>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__Reset_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Reset_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for Reset_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__Reset_Response__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__Reset_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Reset_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__Reset_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Reset_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Reset_Response where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/Reset_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset_Response() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__init(msg: *mut SetSlamPose_Request) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Request>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Request>);
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<SetSlamPose_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Request>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetSlamPose_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: geometry_msgs::msg::rmw::Pose,

}



impl Default for SetSlamPose_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for SetSlamPose_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for SetSlamPose_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for SetSlamPose_Request where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/SetSlamPose_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request() }
  }
}


#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response() -> *const std::ffi::c_void;
}

#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_generator_c")]
extern "C" {
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__init(msg: *mut SetSlamPose_Response) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Response>, size: usize) -> bool;
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Response>);
    fn isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<SetSlamPose_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<SetSlamPose_Response>) -> bool;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetSlamPose_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for SetSlamPose_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__init(&mut msg as *mut _) {
        panic!("Call to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for SetSlamPose_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for SetSlamPose_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for SetSlamPose_Response where Self: Sized {
  const TYPE_NAME: &'static str = "isaac_ros_visual_slam_interfaces/srv/SetSlamPose_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response() }
  }
}






#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath() -> *const std::ffi::c_void;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__FilePath
#[allow(missing_docs, non_camel_case_types)]
pub struct FilePath;

impl rosidl_runtime_rs::Service for FilePath {
    type Request = FilePath_Request;
    type Response = FilePath_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__FilePath() }
    }
}




#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses() -> *const std::ffi::c_void;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__GetAllPoses
#[allow(missing_docs, non_camel_case_types)]
pub struct GetAllPoses;

impl rosidl_runtime_rs::Service for GetAllPoses {
    type Request = GetAllPoses_Request;
    type Response = GetAllPoses_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__GetAllPoses() }
    }
}




#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap() -> *const std::ffi::c_void;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap
#[allow(missing_docs, non_camel_case_types)]
pub struct LocalizeInMap;

impl rosidl_runtime_rs::Service for LocalizeInMap {
    type Request = LocalizeInMap_Request;
    type Response = LocalizeInMap_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__LocalizeInMap() }
    }
}




#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset() -> *const std::ffi::c_void;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__Reset
#[allow(missing_docs, non_camel_case_types)]
pub struct Reset;

impl rosidl_runtime_rs::Service for Reset {
    type Request = Reset_Request;
    type Response = Reset_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__Reset() }
    }
}




#[link(name = "isaac_ros_visual_slam_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose() -> *const std::ffi::c_void;
}

// Corresponds to isaac_ros_visual_slam_interfaces__srv__SetSlamPose
#[allow(missing_docs, non_camel_case_types)]
pub struct SetSlamPose;

impl rosidl_runtime_rs::Service for SetSlamPose {
    type Request = SetSlamPose_Request;
    type Response = SetSlamPose_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__isaac_ros_visual_slam_interfaces__srv__SetSlamPose() }
    }
}


