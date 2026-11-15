#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};




// Corresponds to isaac_ros_visual_slam_interfaces__srv__FilePath_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FilePath_Request {
    /// Request
    pub file_path: std::string::String,

}



impl Default for FilePath_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::FilePath_Request::default())
  }
}

impl rosidl_runtime_rs::Message for FilePath_Request {
  type RmwMsg = super::srv::rmw::FilePath_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        file_path: msg.file_path.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        file_path: msg.file_path.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      file_path: msg.file_path.to_string(),
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__FilePath_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FilePath_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for FilePath_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::FilePath_Response::default())
  }
}

impl rosidl_runtime_rs::Message for FilePath_Response {
  type RmwMsg = super::srv::rmw::FilePath_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GetAllPoses_Request {
    /// Max number of poses to query
    pub max_count: i32,

}



impl Default for GetAllPoses_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::GetAllPoses_Request::default())
  }
}

impl rosidl_runtime_rs::Message for GetAllPoses_Request {
  type RmwMsg = super::srv::rmw::GetAllPoses_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        max_count: msg.max_count,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      max_count: msg.max_count,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      max_count: msg.max_count,
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__GetAllPoses_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GetAllPoses_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

    /// List of optimized poses
    pub poses: Vec<geometry_msgs::msg::PoseStamped>,

}



impl Default for GetAllPoses_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::GetAllPoses_Response::default())
  }
}

impl rosidl_runtime_rs::Message for GetAllPoses_Response {
  type RmwMsg = super::srv::rmw::GetAllPoses_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
        poses: msg.poses
          .into_iter()
          .map(|elem| geometry_msgs::msg::PoseStamped::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
        poses: msg.poses
          .iter()
          .map(|elem| geometry_msgs::msg::PoseStamped::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
      poses: msg.poses
          .into_iter()
          .map(geometry_msgs::msg::PoseStamped::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LocalizeInMap_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub map_folder_path: std::string::String,

    /// Pose hint used to localize in the map. Pose hint corresponds to map_T_robot.
    pub pose_hint: geometry_msgs::msg::Pose,

}



impl Default for LocalizeInMap_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::LocalizeInMap_Request::default())
  }
}

impl rosidl_runtime_rs::Message for LocalizeInMap_Request {
  type RmwMsg = super::srv::rmw::LocalizeInMap_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        map_folder_path: msg.map_folder_path.as_str().into(),
        pose_hint: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.pose_hint)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        map_folder_path: msg.map_folder_path.as_str().into(),
        pose_hint: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose_hint)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      map_folder_path: msg.map_folder_path.to_string(),
      pose_hint: geometry_msgs::msg::Pose::from_rmw_message(msg.pose_hint),
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__LocalizeInMap_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LocalizeInMap_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for LocalizeInMap_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::LocalizeInMap_Response::default())
  }
}

impl rosidl_runtime_rs::Message for LocalizeInMap_Response {
  type RmwMsg = super::srv::rmw::LocalizeInMap_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__Reset_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Reset_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub structure_needs_at_least_one_member: u8,

}



impl Default for Reset_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::Reset_Request::default())
  }
}

impl rosidl_runtime_rs::Message for Reset_Request {
  type RmwMsg = super::srv::rmw::Reset_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        structure_needs_at_least_one_member: msg.structure_needs_at_least_one_member,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      structure_needs_at_least_one_member: msg.structure_needs_at_least_one_member,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      structure_needs_at_least_one_member: msg.structure_needs_at_least_one_member,
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__Reset_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Reset_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for Reset_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::Reset_Response::default())
  }
}

impl rosidl_runtime_rs::Message for Reset_Response {
  type RmwMsg = super::srv::rmw::Reset_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetSlamPose_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: geometry_msgs::msg::Pose,

}



impl Default for SetSlamPose_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetSlamPose_Request::default())
  }
}

impl rosidl_runtime_rs::Message for SetSlamPose_Request {
  type RmwMsg = super::srv::rmw::SetSlamPose_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      pose: geometry_msgs::msg::Pose::from_rmw_message(msg.pose),
    }
  }
}


// Corresponds to isaac_ros_visual_slam_interfaces__srv__SetSlamPose_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetSlamPose_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,

}



impl Default for SetSlamPose_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetSlamPose_Response::default())
  }
}

impl rosidl_runtime_rs::Message for SetSlamPose_Response {
  type RmwMsg = super::srv::rmw::SetSlamPose_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
    }
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


