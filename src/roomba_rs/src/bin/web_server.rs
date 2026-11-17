//! web_server (Rust): drop-in replacement for roomba_web/server.py (see its docstring for the
//! routes). Same node name, parameters, topics and QoS, NavigateToPose action and /roomba/set_mode
//! clients, TF, HTTP routes and status codes, websocket JSON (10 Hz), MJPEG stream, static files
//! (share/roomba_web/www) and login (the same ~/.config/roomba_web/auth.json and session cookie).
//! Rust only: /depth.mjpg, the depth camera colourised (turbo, red near to blue far, black where
//! there is no depth), at most depth_fps frames a second and only while someone watches it.
//! The ROS side runs on rclrs' executor thread (like the Python node's spin thread, including the
//! 10 Hz drive pad deadman), the HTTP server on tokio (hyper + tokio-tungstenite). The logic lives
//! in roomba_rs::web (state, request validation, images, auth, aiohttp behaviour).

use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{anyhow, Context as _, Result};
use base64::Engine as _;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Empty, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{self, HeaderMap};
use hyper::{Method, Request, Response};
use hyper_util::rt::{TokioIo, TokioTimer};
use isaac_ros_visual_slam_interfaces::msg::VisualSlamStatus;
use rclrs::*;
use ros_env::{geometry_msgs, nav2_msgs, nav_msgs, roomba_msgs, sensor_msgs, std_msgs, vision_msgs};
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::protocol::{Role, WebSocketConfig};
use tokio_tungstenite::tungstenite::{handshake::derive_accept_key, Message};
use tokio_tungstenite::WebSocketStream;

use roomba_rs::tf_listener::{stamp_ns, TfListener};
use roomba_rs::web::auth::{self, AuthPaths, Sessions, SESSION_COOKIE};
use roomba_rs::web::http::{self, Conditional, MimeTypes, Reply};
use roomba_rs::web::json::{self, Py};
use roomba_rs::web::state::{self, Core, Detection, Grid, MAX_BODY, MAX_STREAMS};
use roomba_rs::web::image;

type NavigateToPose = nav2_msgs::action::NavigateToPose;
type Body = UnsyncBoxBody<Bytes, Infallible>;

/// time.monotonic(): CLOCK_MONOTONIC in seconds.
fn mono() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime only writes the timespec passed to it.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Declares parameters and keeps their handles (rclrs undeclares a parameter when its handle drops).
struct Declared<'a> {
    node: &'a Node,
    handles: Vec<Box<dyn std::any::Any>>,
}

impl Declared<'_> {
    fn get<T: ParameterVariant + 'static>(&mut self, name: &str, default: T) -> Result<T> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        let value = p.get();
        self.handles.push(Box::new(p));
        Ok(value)
    }
}

enum NavRequest {
    Goal(f64, f64),
    Cancel,
}

enum RobotRequest {
    Stop,
    Mode(u8),
}

/// RosState: the node's ROS objects and the shared state (Core).
struct Ros {
    node: Node,
    core: Mutex<Core>,
    tf: TfListener,
    map_frame: String,
    base_frame: String,
    nav_client: ActionClient<NavigateToPose>,
    mode_client: Client<roomba_msgs::srv::SetMode>,
    pub_cmd: Publisher<geometry_msgs::msg::Twist>,
    pub_cmd_nav: Publisher<geometry_msgs::msg::Twist>,
    pub_zone_cmd: Publisher<std_msgs::msg::String>,
    pub_explore: Publisher<std_msgs::msg::String>,
    pub_floor_grid: Publisher<std_msgs::msg::Bool>,
    nav_requests: Mutex<VecDeque<NavRequest>>,
    robot_requests: Mutex<VecDeque<RobotRequest>>,
    nav_handle: Mutex<Option<CancellationClient<NavigateToPose>>>,
    drive_active: Mutex<bool>,
    /// Latest camera JPEG (annotated detector frame or plain preview) for the MJPEG streams.
    jpeg: tokio::sync::watch::Sender<Option<Bytes>>,
    last_annotated: Mutex<f64>,
    last_preview: Mutex<f64>,
    preview_sub: Mutex<Option<Subscription<sensor_msgs::msg::rmw::Image>>>,
    preview_fps: f64,
    preview_topic: String,
    preview_width: usize,
    /// Depth view: JPEGs for /depth.mjpg, its viewers (the subscription exists only while there
    /// are any), the colour table and the frame rate.
    depth_jpeg: tokio::sync::watch::Sender<Option<Bytes>>,
    depth_viewers: AtomicUsize,
    depth_sub: Mutex<Option<Subscription<sensor_msgs::msg::rmw::Image>>>,
    last_depth: Mutex<f64>,
    depth_topic: String,
    depth_fps: f64,
    depth_colours: image::DepthColours,
    depth_range_mm: (u32, u32),
    rt: tokio::runtime::Handle,
}

fn grid_of(msg: &nav_msgs::msg::rmw::OccupancyGrid) -> (usize, usize, f32, [f64; 2], [f64; 4]) {
    let i = &msg.info;
    let (p, q) = (&i.origin.position, &i.origin.orientation);
    (i.width as usize, i.height as usize, i.resolution, [p.x, p.y], [q.x, q.y, q.z, q.w])
}

impl Ros {
    // ---- ROS callbacks (executor thread)

    fn on_jpeg(&self, msg: sensor_msgs::msg::rmw::CompressedImage) {
        *lock(&self.last_annotated) = mono();
        self.jpeg.send_replace(Some(Bytes::copy_from_slice(&msg.data)));
    }

    fn preview_check(self: &Arc<Self>) {
        let detector_live = mono() - *lock(&self.last_annotated) < 3.0;
        let mut sub = lock(&self.preview_sub);
        if detector_live && sub.is_some() {
            *sub = None;
        } else if !detector_live && sub.is_none() {
            let ros = Arc::clone(self);
            match self
                .node
                .create_subscription::<sensor_msgs::msg::rmw::Image, _>(self.preview_topic.as_str().sensor_data_qos(), move |m: sensor_msgs::msg::rmw::Image| ros.on_preview(m))
            {
                Ok(s) => *sub = Some(s),
                Err(e) => log!(self.node.error(), "cannot subscribe to {}: {e}", self.preview_topic),
            }
        }
    }

    fn on_preview(&self, msg: sensor_msgs::msg::rmw::Image) {
        let now = mono();
        if now - *lock(&self.last_preview) < 1.0 / self.preview_fps || now - *lock(&self.last_annotated) < 3.0 {
            return;
        }
        let encoding = msg.encoding.to_string();
        if encoding != "mono8" && encoding != "8UC1" {
            return;
        }
        *lock(&self.last_preview) = now;
        if let Some(jpg) = image::preview_jpeg(&msg.data, msg.width as usize, msg.height as usize, msg.step as usize, self.preview_width) {
            self.jpeg.send_replace(Some(Bytes::from(jpg)));
        }
    }

    /// Subscribes to the depth image while /depth.mjpg has viewers.
    fn depth_check(self: &Arc<Self>) {
        let watched = self.depth_viewers.load(Ordering::SeqCst) > 0;
        let mut sub = lock(&self.depth_sub);
        if !watched && sub.is_some() {
            *sub = None;
            self.depth_jpeg.send_replace(None);
        } else if watched && sub.is_none() {
            let ros = Arc::clone(self);
            match self
                .node
                .create_subscription::<sensor_msgs::msg::rmw::Image, _>(self.depth_topic.as_str().sensor_data_qos(), move |m: sensor_msgs::msg::rmw::Image| ros.on_depth(m))
            {
                Ok(s) => *sub = Some(s),
                Err(e) => log!(self.node.error(), "cannot subscribe to {}: {e}", self.depth_topic),
            }
        }
    }

    fn on_depth(&self, msg: sensor_msgs::msg::rmw::Image) {
        let now = mono();
        if now - *lock(&self.last_depth) < 1.0 / self.depth_fps || self.depth_viewers.load(Ordering::SeqCst) == 0 {
            return;
        }
        let encoding = msg.encoding.to_string();
        if encoding != "16UC1" && encoding != "mono16" {
            return;
        }
        *lock(&self.last_depth) = now;
        let (w, h, step) = (msg.width as usize, msg.height as usize, msg.step as usize);
        if let Some(jpg) = self.depth_colours.jpeg(&msg.data, w, h, step, msg.is_bigendian != 0, 75) {
            self.depth_jpeg.send_replace(Some(Bytes::from(jpg)));
        }
    }

    /// _to_map: map_frame <- frame at the header stamp, else the latest.
    fn to_map(&self, header: &std_msgs::msg::Header) -> Option<roomba_rs::tf_buffer::Transform> {
        [stamp_ns(&header.stamp), 0]
            .into_iter()
            .find_map(|t| self.tf.lookup_transform(&self.map_frame, &header.frame_id, t).ok())
            .map(|t| t.transform)
    }

    fn on_dets(&self, msg: vision_msgs::msg::Detection3DArray, excluded: bool) {
        let tf = if msg.detections.is_empty() { None } else { self.to_map(&msg.header) };
        let items = msg
            .detections
            .iter()
            .map(|d| {
                let c = &d.bbox.center.position;
                let hyp = d.results.first().map(|r| (r.hypothesis.class_id.as_str(), r.hypothesis.score));
                Detection::new(hyp, c.z, &d.id, tf.map(|t| t.apply([c.x, c.y, c.z])))
            })
            .collect();
        lock(&self.core).on_dets(excluded, mono(), items);
    }

    fn on_map(&self, msg: nav_msgs::msg::rmw::OccupancyGrid, filled: bool) {
        if !lock(&self.core).take_map(mono(), filled) {
            return;
        }
        let (width, height, resolution, origin, orientation) = grid_of(&msg);
        let frame = msg.header.frame_id.to_string();
        let grid = Grid { width, height, resolution, origin, orientation, frame: &frame, data: &msg.data };
        if let Some(png) = image::map_png(width, height, &msg.data) {
            lock(&self.core).set_map(&grid, png);
        }
    }

    fn on_costmap(&self, msg: nav_msgs::msg::rmw::OccupancyGrid) {
        let (width, height, resolution, origin, orientation) = grid_of(&msg);
        let frame = msg.header.frame_id.to_string();
        let Ok(tf) = self.tf.lookup_transform(&self.map_frame, &frame, 0) else { return };
        let grid = Grid { width, height, resolution, origin, orientation, frame: &frame, data: &msg.data };
        if let Some(points) = state::costmap_points(&grid, &tf.transform) {
            lock(&self.core).set_obstacles(points);
        }
    }

    fn path_xy(msg: &nav_msgs::msg::rmw::Path) -> Vec<[f64; 2]> {
        msg.poses.iter().map(|p| [p.pose.position.x, p.pose.position.y]).collect()
    }

    fn on_battery(&self, msg: sensor_msgs::msg::BatteryState) {
        lock(&self.core).on_battery(msg.percentage, msg.voltage, msg.power_supply_status, mono());
    }

    fn on_sensors(&self, m: roomba_msgs::msg::SensorState) {
        let flags = [
            m.bump_left,
            m.bump_right,
            m.wheel_drop_left,
            m.wheel_drop_right,
            m.cliff_left,
            m.cliff_front_left,
            m.cliff_front_right,
            m.cliff_right,
            m.home_base_available,
            m.internal_charger_available,
        ];
        lock(&self.core).on_sensors(flags, m.oi_mode, mono());
    }

    // ---- robot and navigation (executor thread)

    fn nav_available(&self) -> bool {
        self.nav_client.server_is_available().unwrap_or(false)
    }

    /// With navigation running, manual driving goes through Nav2's collision monitor
    /// (cmd_vel_nav -> collision_monitor -> cmd_vel) so its stop zone also protects it.
    fn publish_drive(&self, v: f64, w: f64) {
        let mut msg = geometry_msgs::msg::Twist::default();
        msg.linear.x = v;
        msg.angular.z = w;
        let publisher = if self.nav_available() { &self.pub_cmd_nav } else { &self.pub_cmd };
        publisher.publish(msg).ok();
    }

    fn publish_explore(&self, cmd: &str) {
        self.pub_explore.publish(std_msgs::msg::String { data: cmd.into() }).ok();
    }

    /// explore_command('stop'): only while the explorer is driving (avoids log noise).
    fn explore_stop(&self) {
        let active = lock(&self.core).exploring();
        if active {
            self.publish_explore("stop");
        }
    }

    /// Forwards the drive pad while it is held, then stops once; handles stop / mode requests.
    fn drive_tick(self: &Arc<Self>) {
        let held = lock(&self.core).held_drive(mono());
        let active = *lock(&self.drive_active);
        if let Some((v, w)) = held {
            if !active {
                *lock(&self.drive_active) = true;
                let navigating = lock(&self.core).nav_goal.is_some();
                if navigating {
                    // manual driving takes over from navigation
                    lock(&self.nav_requests).push_back(NavRequest::Cancel);
                }
                self.explore_stop();
            }
            self.publish_drive(v, w);
        } else if active {
            *lock(&self.drive_active) = false;
            self.publish_drive(0.0, 0.0);
        }
        loop {
            let request = lock(&self.robot_requests).pop_front();
            match request {
                None => break,
                Some(RobotRequest::Stop) => {
                    lock(&self.core).drive = None;
                    *lock(&self.drive_active) = false;
                    self.explore_stop();
                    self.publish_drive(0.0, 0.0);
                }
                Some(RobotRequest::Mode(mode)) => self.set_mode(mode),
            }
        }
    }

    fn set_mode(self: &Arc<Self>, mode: u8) {
        let request = roomba_msgs::srv::SetMode_Request { mode };
        let promise = self.mode_client.call::<_, roomba_msgs::srv::SetMode_Response>(request);
        let ros = Arc::clone(self);
        self.rt.spawn(async move {
            let text = match promise {
                Ok(p) => match p.await {
                    Ok(res) if res.success => res.message,
                    Ok(res) => format!("Could not change mode: {}", res.message),
                    Err(e) => format!("Could not change mode: {e}"),
                },
                Err(e) => format!("Could not change mode: {e}"),
            };
            lock(&ros.core).on_mode_result(mono(), text);
        });
    }

    /// Action calls happen here, not on the web side.
    fn process_nav_requests(self: &Arc<Self>) {
        loop {
            let request = lock(&self.nav_requests).pop_front();
            match request {
                None => return,
                Some(NavRequest::Goal(x, y)) => {
                    self.explore_stop(); // a goal from the map takes over from exploring
                    let mut goal = nav2_msgs::action::NavigateToPose_Goal::default();
                    goal.pose.header.frame_id = self.map_frame.clone();
                    goal.pose.header.stamp = self.node.get_clock().now().to_ros_msg().unwrap_or_default();
                    goal.pose.pose.position.x = x;
                    goal.pose.pose.position.y = y;
                    goal.pose.pose.orientation.w = 1.0;
                    lock(&self.core).nav_goal = Some([x, y]);
                    match self.nav_client.try_request_goal(goal) {
                        Ok(requested) => {
                            let ros = Arc::clone(self);
                            self.rt.spawn(async move {
                                match requested.await {
                                    Some(handle) => *lock(&ros.nav_handle) = Some(handle.cancellation),
                                    None => {
                                        lock(&ros.core).nav_goal = None;
                                        log!(ros.node.warn(), "Navigation goal rejected");
                                    }
                                }
                            });
                        }
                        Err(e) => log!(self.node.error(), "Cannot send the navigation goal: {e}"),
                    }
                }
                Some(NavRequest::Cancel) => {
                    lock(&self.core).nav_goal = None;
                    if let Some(handle) = lock(&self.nav_handle).as_ref() {
                        let _ = handle.try_cancel();
                    }
                }
            }
        }
    }

    // ---- web side

    fn pose(&self) -> Py {
        match self.tf.lookup_transform(&self.map_frame, &self.base_frame, 0) {
            Ok(t) => state::pose_py(t.transform.translation, t.transform.rotation),
            Err(_) => Py::None,
        }
    }

    fn snapshot_json(&self) -> String {
        let now = mono();
        let nav_available = self.nav_available();
        let pose = self.pose();
        let explore_subscribed = self.pub_explore.get_subscription_count().unwrap_or(0) > 0;
        let state = lock(&self.core).snapshot(now, nav_available, pose, explore_subscribed);
        json::dumps(&state)
    }
}

// ---------------------------------------------------------------- HTTP

struct App {
    ros: Arc<Ros>,
    www: PathBuf,
    mime: MimeTypes,
    auth_paths: AuthPaths,
    sessions: Mutex<Sessions>,
    /// scrypt needs 16 MiB per check: at most as many at once as Python's default thread pool.
    scrypt_slots: tokio::sync::Semaphore,
    streams: AtomicUsize,
}

/// A handler result: a buffered reply, or a streaming / upgrade response.
enum Out {
    Reply(Reply),
    Raw(Response<Body>),
}

fn full(body: Vec<u8>) -> Body {
    Full::new(Bytes::from(body)).boxed_unsync()
}

fn to_response(r: Reply) -> Response<Body> {
    let mut builder = Response::builder().status(r.status);
    for (name, value) in &r.headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let reason = http::reason(r.status);
    if !reason.is_empty() {
        builder = builder.extension(hyper::ext::ReasonPhrase::from_static(reason.as_bytes()));
    }
    builder.body(full(r.body)).unwrap_or_else(|_| to_response(Reply::internal_error()))
}

fn header_str(headers: &HeaderMap, name: header::HeaderName) -> Option<String> {
    headers.get(name).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
}

/// 405 with aiohttp's Allow header, or None when the method is routed.
fn method_not_allowed(method: &Method, allowed: &[&str]) -> Option<Reply> {
    if allowed.contains(&method.as_str()) {
        return None;
    }
    let mut sorted: Vec<&str> = allowed.to_vec();
    sorted.sort_unstable();
    Some(Reply::error(405).with("Allow", &sorted.join(",")))
}

const GET: &[&str] = &["GET", "HEAD"];
const POST: &[&str] = &["POST"];

/// The body, refused (413) at MAX_BODY bytes like aiohttp's client_max_size.
async fn read_body(body: &mut Incoming) -> Result<Vec<u8>, Reply> {
    let mut out = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| Reply::error(400))?;
        if let Ok(data) = frame.into_data() {
            out.extend_from_slice(&data);
            if out.len() >= MAX_BODY {
                return Err(Reply::text(
                    413,
                    &format!("Maximum request body size {MAX_BODY} exceeded, actual body size {}", out.len()),
                ));
            }
        }
    }
    Ok(out)
}

/// Python's repr() of an optional header value (for the websocket handshake errors).
fn py_repr_opt(v: Option<&str>) -> String {
    v.map_or("None".into(), json::repr_str)
}

/// MJPEG over HTTP: every new JPEG from `rx` as a part; `slot` is held until the client leaves.
fn mjpeg(rx: tokio::sync::watch::Receiver<Option<Bytes>>, slot: Box<dyn Send>, method: &Method, extra: &[(&str, String)]) -> Out {
    let mut builder = Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame")
        .header(header::CACHE_CONTROL, "no-cache, no-store");
    for (name, value) in extra {
        builder = builder.header(*name, value.as_str());
    }
    if method == Method::HEAD {
        drop(slot);
        return Out::Raw(builder.body(Empty::new().boxed_unsync()).unwrap_or_else(|_| to_response(Reply::internal_error())));
    }
    let frames = futures_util::stream::unfold((rx, slot, true), |(mut rx, slot, mut first)| async move {
        loop {
            if !first && rx.changed().await.is_err() {
                return None;
            }
            first = false;
            let jpg = rx.borrow_and_update().clone();
            if let Some(jpg) = jpg {
                let mut part = Vec::with_capacity(jpg.len() + 80);
                part.extend_from_slice(format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpg.len()).as_bytes());
                part.extend_from_slice(&jpg);
                part.extend_from_slice(b"\r\n");
                return Some((Ok::<_, Infallible>(Frame::data(Bytes::from(part))), (rx, slot, first)));
            }
        }
    });
    Out::Raw(builder.body(BodyExt::boxed_unsync(StreamBody::new(frames))).unwrap_or_else(|_| to_response(Reply::internal_error())))
}

struct DepthSlot(Arc<Ros>);

impl Drop for DepthSlot {
    fn drop(&mut self) {
        self.0.depth_viewers.fetch_sub(1, Ordering::SeqCst);
    }
}

struct StreamSlot(Arc<App>);

impl Drop for StreamSlot {
    fn drop(&mut self) {
        self.0.streams.fetch_sub(1, Ordering::SeqCst);
    }
}

impl App {
    fn authed(&self, headers: &HeaderMap) -> bool {
        let token = http::cookie(header_str(headers, header::COOKIE).as_deref(), SESSION_COOKIE);
        lock(&self.sessions).valid(token.as_deref(), http::wall_time())
    }

    async fn handle(self: Arc<Self>, mut req: Request<Incoming>) -> Response<Body> {
        let raw_path = req.uri().path().to_string();
        let path = http::unquote_path(&raw_path);
        let out = self.dispatch(&mut req, &raw_path, &path).await;
        let mut resp = match out {
            Out::Reply(r) => to_response(r),
            Out::Raw(r) => r,
        };
        // Browsers kept running an old app.js after an update: static files always revalidate.
        if path.starts_with("/static/") {
            resp.headers_mut().insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
        }
        resp
    }

    async fn dispatch(self: &Arc<Self>, req: &mut Request<Incoming>, raw_path: &str, path: &str) -> Out {
        // The guard middleware: everything except the login page and static files needs a session.
        if !(path == "/login" || path == "/favicon.ico" || path.starts_with("/static/")) {
            if !self.authed(req.headers()) {
                if path.starts_with("/api/") || path == "/ws" || path == "/stream.mjpg" || path == "/depth.mjpg" {
                    return Out::Reply(Reply::error(401));
                }
                return Out::Reply(Reply::found("/login"));
            }
            let m = req.method();
            if (m == Method::POST || m == Method::DELETE || m == Method::PUT)
                && path.starts_with("/api/")
                && req.headers().get("X-Requested-With").map(|v| v.as_bytes()) != Some(state::CSRF_VALUE.as_bytes())
            {
                return Out::Reply(Reply::text(403, "missing request header"));
            }
        }
        let method = req.method().clone();
        let allowed: &[&str] = match raw_path {
            "/" | "/stream.mjpg" | "/depth.mjpg" | "/ws" | "/api/state" | "/api/map.json" | "/api/map.png" | "/api/path" | "/api/plan" | "/api/obstacles" => GET,
            "/login" | "/api/zones" => &["GET", "HEAD", "POST"],
            "/logout" | "/api/goal" | "/api/goal/cancel" | "/api/drive" | "/api/robot/mode" | "/api/robot/stop" | "/api/explore" | "/api/floor_grid" => POST,
            p if p.strip_prefix("/api/zones/").is_some_and(|n| !n.is_empty() && !n.contains(['/', '{', '}'])) => &["DELETE"],
            p if p == "/static" || p.starts_with("/static/") => GET,
            _ => return Out::Reply(Reply::error(404)),
        };
        if let Some(r) = method_not_allowed(&method, allowed) {
            return Out::Reply(r);
        }
        let ros = &self.ros;
        let charset = http::charset(header_str(req.headers(), header::CONTENT_TYPE).as_deref());
        let charset = charset.as_deref();
        macro_rules! body {
            () => {
                match read_body(req.body_mut()).await {
                    Ok(b) => b,
                    Err(r) => return Out::Reply(r),
                }
            };
        }
        macro_rules! parsed {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(r) => return Out::Reply(r),
                }
            };
        }
        let reply = match (raw_path, method == Method::POST || method == Method::DELETE) {
            ("/", _) => self.page("index.html", req.headers(), &method),
            ("/login", false) => self.page("login.html", req.headers(), &method),
            ("/login", true) => return self.login(req).await,
            ("/logout", _) => {
                let token = http::cookie(header_str(req.headers(), header::COOKIE).as_deref(), SESSION_COOKIE);
                lock(&self.sessions).revoke(token.as_deref());
                Reply::found("/login").with("Set-Cookie", &format!("{SESSION_COOKIE}=\"\"; expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0; Path=/"))
            }
            ("/stream.mjpg", _) => return self.stream(&method),
            ("/depth.mjpg", _) => return self.depth_stream(&method),
            ("/ws", _) => return self.websocket(req),
            ("/api/state", _) => Reply::json(ros.snapshot_json()),
            ("/api/map.json", _) => match lock(&ros.core).map_json() {
                Some(meta) => Reply::json(meta),
                None => Reply::text(404, "no map yet"),
            },
            ("/api/map.png", _) => {
                let png = lock(&ros.core).map_png.clone();
                match png {
                    Some(png) => Reply::new(200, "image/png", png.as_ref().clone()).with("Cache-Control", "no-cache"),
                    None => Reply::text(404, "no map yet"),
                }
            }
            ("/api/path", _) => Reply::json(lock(&ros.core).path_json()),
            ("/api/zones", false) => Reply::json(lock(&ros.core).zones_json()),
            ("/api/zones", true) => {
                let body = body!();
                let (name, pts, kind) = parsed!(state::parse_add_zone(&body, charset));
                match lock(&ros.core).zone_exists(&name) {
                    Ok(false) => {}
                    Ok(true) => return Out::Reply(state::zone_conflict(&name)),
                    Err(()) => return Out::Reply(Reply::internal_error()),
                }
                let data = state::add_zone_command(name, &pts, &kind);
                ros.pub_zone_cmd.publish(std_msgs::msg::String { data }).ok();
                state::ok_json()
            }
            ("/api/goal", _) => {
                let body = body!();
                let [x, y] = parsed!(state::parse_goal(&body, charset));
                if !ros.nav_available() {
                    return Out::Reply(Reply::text(503, "navigation is not running (start ~/run_nav.sh)"));
                }
                lock(&ros.nav_requests).push_back(NavRequest::Goal(x, y));
                state::ok_json()
            }
            ("/api/goal/cancel", _) => {
                lock(&ros.nav_requests).push_back(NavRequest::Cancel);
                state::ok_json()
            }
            ("/api/plan", _) => Reply::json(lock(&ros.core).plan_json()),
            ("/api/obstacles", _) => Reply::json(lock(&ros.core).obstacles_json()),
            ("/api/drive", _) => {
                let body = body!();
                let [v, w] = parsed!(state::parse_drive(&body, charset));
                let mut core = lock(&ros.core);
                if !core.robot_available(mono()) {
                    return Out::Reply(Reply::text(503, "the robot driver is not running"));
                }
                core.set_drive(v, w, mono());
                state::ok_json()
            }
            ("/api/robot/mode", _) => {
                let body = body!();
                let mode = parsed!(state::parse_mode(&body, charset));
                if !ros.mode_client.service_is_ready().unwrap_or(false) {
                    return Out::Reply(Reply::text(503, "the robot driver is not running"));
                }
                lock(&ros.robot_requests).push_back(RobotRequest::Mode(mode));
                state::ok_json()
            }
            ("/api/robot/stop", _) => {
                lock(&ros.robot_requests).push_back(RobotRequest::Stop);
                lock(&ros.nav_requests).push_back(NavRequest::Cancel);
                state::ok_json()
            }
            ("/api/explore", _) => {
                let body = body!();
                let action = parsed!(state::parse_explore(&body, charset));
                if ros.pub_explore.get_subscription_count().unwrap_or(0) == 0 {
                    return Out::Reply(Reply::text(503, "exploration is not running (robot.launch.py with nav)"));
                }
                if action == "start" {
                    lock(&ros.nav_requests).push_back(NavRequest::Cancel); // don't fight a goal sent from the map
                }
                ros.publish_explore(&action);
                state::ok_json()
            }
            ("/api/floor_grid", _) => {
                let body = body!();
                let on = parsed!(state::parse_floor_grid(&body, charset));
                lock(&ros.core).floor_grid = on;
                ros.pub_floor_grid.publish(std_msgs::msg::Bool { data: on }).ok();
                Reply::json(format!("{{\"ok\": true, \"on\": {on}}}"))
            }
            (p, _) if p.starts_with("/api/zones/") => {
                let name = http::unquote_path(&p["/api/zones/".len()..]);
                let data = state::delete_zone_command(&name);
                ros.pub_zone_cmd.publish(std_msgs::msg::String { data }).ok();
                state::ok_json()
            }
            (p, _) => {
                let filename = http::unquote_path(p.get("/static/".len()..).unwrap_or(""));
                match http::static_file(&self.www, &filename) {
                    Ok(file) => http::file_response(&file, &conditional(req.headers(), &method), &self.mime),
                    Err(status) => Reply::error(status),
                }
            }
        };
        Out::Reply(reply)
    }

    /// FileResponse for the app and login pages (no-cache: the page always revalidates).
    fn page(&self, name: &str, headers: &HeaderMap, method: &Method) -> Reply {
        let mut r = http::file_response(&self.www.join(name), &conditional(headers, method), &self.mime);
        r.set("Cache-Control", "no-cache");
        r
    }

    async fn login(self: &Arc<Self>, req: &mut Request<Incoming>) -> Out {
        let content_type = header_str(req.headers(), header::CONTENT_TYPE);
        let body = match read_body(req.body_mut()).await {
            Ok(b) => b,
            Err(r) => return Out::Reply(r),
        };
        let password = match form_password(content_type.as_deref(), body).await {
            Ok(p) => p,
            Err(r) => return Out::Reply(r),
        };
        let ok = {
            let _slot = self.scrypt_slots.acquire().await;
            let paths = self.auth_paths.clone();
            tokio::task::spawn_blocking(move || auth::check_password(&paths, &password)).await
        };
        let ok = match ok {
            Ok(Ok(ok)) => ok,
            _ => {
                log!(self.ros.node.error(), "Cannot read {}", self.auth_paths.auth_file.display());
                false
            }
        };
        if !ok {
            tokio::time::sleep(Duration::from_secs(1)).await; // slow down guessing
            return Out::Reply(Reply::found("/login?error=1"));
        }
        let token = lock(&self.sessions).new_session(http::wall_time());
        let cookie = format!("{SESSION_COOKIE}={token}; HttpOnly; Max-Age={}; Path=/; SameSite=Strict", auth::SESSION_TTL_S as u64);
        Out::Reply(Reply::found("/").with("Set-Cookie", &cookie))
    }

    fn stream(self: &Arc<Self>, method: &Method) -> Out {
        let taken = self.streams.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < MAX_STREAMS).then_some(n + 1));
        if taken.is_err() {
            return Out::Reply(Reply::text(503, "too many camera viewers"));
        }
        let slot = StreamSlot(Arc::clone(self));
        mjpeg(self.ros.jpeg.subscribe(), Box::new(slot), method, &[])
    }

    /// The colourised depth camera. X-Depth-Range (mm) is for the page's legend.
    fn depth_stream(self: &Arc<Self>, method: &Method) -> Out {
        let ros = &self.ros;
        let taken = ros.depth_viewers.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < MAX_STREAMS).then_some(n + 1));
        if taken.is_err() {
            return Out::Reply(Reply::text(503, "too many depth viewers"));
        }
        let slot = DepthSlot(Arc::clone(ros));
        let range = format!("{}-{}", ros.depth_range_mm.0, ros.depth_range_mm.1);
        mjpeg(ros.depth_jpeg.subscribe(), Box::new(slot), method, &[("X-Depth-Range", range)])
    }
    /// aiohttp's WebSocketResponse handshake checks, then the 10 Hz state push.
    fn websocket(self: &Arc<Self>, req: &mut Request<Incoming>) -> Out {
        let h = req.headers();
        let upgrade = header_str(h, header::UPGRADE);
        if upgrade.as_deref().unwrap_or("").to_lowercase().trim() != "websocket" {
            let text = format!("No WebSocket UPGRADE hdr: {}\n Can \"Upgrade\" only to \"WebSocket\".", upgrade.as_deref().unwrap_or("None"));
            return Out::Reply(Reply::text(400, &text));
        }
        let connection = header_str(h, header::CONNECTION);
        if !connection.as_deref().unwrap_or("").to_lowercase().contains("upgrade") {
            return Out::Reply(Reply::text(400, &format!("No CONNECTION upgrade hdr: {}", connection.as_deref().unwrap_or("None"))));
        }
        let version = header_str(h, header::SEC_WEBSOCKET_VERSION).unwrap_or_default();
        if !matches!(version.as_str(), "13" | "8" | "7") {
            return Out::Reply(Reply::text(400, &format!("Unsupported version: {version}")));
        }
        let key = header_str(h, header::SEC_WEBSOCKET_KEY);
        let key_ok = key.as_deref().is_some_and(|k| {
            let cleaned: String = k.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')).collect();
            base64::engine::general_purpose::STANDARD.decode(cleaned).is_ok_and(|b| b.len() == 16)
        });
        let Some(key) = key.filter(|_| key_ok) else {
            return Out::Reply(Reply::text(400, &format!("Handshake error: {}", py_repr_opt(header_str(h, header::SEC_WEBSOCKET_KEY).as_deref()))));
        };
        let accept = derive_accept_key(key.as_bytes());
        let on_upgrade = hyper::upgrade::on(req);
        let app = Arc::clone(self);
        tokio::spawn(async move {
            if let Ok(upgraded) = on_upgrade.await {
                let config = WebSocketConfig::default().max_message_size(Some(4 << 20)).max_frame_size(Some(4 << 20));
                let ws = WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, Some(config)).await;
                app.run_websocket(ws).await;
            }
        });
        let resp = Response::builder()
            .status(101)
            .header(header::UPGRADE, "websocket")
            .header(header::CONNECTION, "upgrade")
            .header(header::SEC_WEBSOCKET_ACCEPT, accept)
            .body(Empty::new().boxed_unsync())
            .unwrap_or_else(|_| to_response(Reply::internal_error()));
        Out::Raw(resp)
    }

    /// Sends the state every 0.1 s (after each send); pings after 10 s without a message from the
    /// client and drops the connection if nothing arrives within 5 s more (aiohttp heartbeat=10).
    async fn run_websocket<S>(self: Arc<Self>, ws: WebSocketStream<S>)
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let (mut tx, mut rx) = ws.split();
        let mut next_send = Instant::now();
        let mut last_rx = Instant::now();
        let mut ping_sent: Option<Instant> = None;
        loop {
            let heartbeat = match ping_sent {
                None => last_rx + Duration::from_secs(10),
                Some(t) => t + Duration::from_secs(5),
            };
            tokio::select! {
                msg = rx.next() => match msg {
                    None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => {
                        last_rx = Instant::now();
                        ping_sent = None;
                    }
                },
                _ = tokio::time::sleep_until(next_send) => {
                    let text = self.ros.snapshot_json();
                    if tx.send(Message::text(text)).await.is_err() {
                        break;
                    }
                    next_send = Instant::now() + Duration::from_millis(100);
                },
                _ = tokio::time::sleep_until(heartbeat) => {
                    if ping_sent.is_some() {
                        return; // no answer: drop the connection
                    }
                    if tx.send(Message::Ping(Bytes::new())).await.is_err() {
                        break;
                    }
                    ping_sent = Some(Instant::now());
                }
            }
        }
        let _ = tx.flush().await;
        let _ = tx.close().await;
    }
}

fn conditional(headers: &HeaderMap, method: &Method) -> Conditional {
    let h = |n: header::HeaderName| header_str(headers, n);
    Conditional {
        head: method == Method::HEAD,
        accept_encoding: h(header::ACCEPT_ENCODING),
        if_match: h(header::IF_MATCH),
        if_none_match: h(header::IF_NONE_MATCH),
        if_modified_since: h(header::IF_MODIFIED_SINCE),
        if_unmodified_since: h(header::IF_UNMODIFIED_SINCE),
        if_range: h(header::IF_RANGE),
        range: h(header::RANGE),
    }
}

/// request.post()['password'] (first value; '' when absent or not a form).
async fn form_password(content_type: Option<&str>, body: Vec<u8>) -> Result<String, Reply> {
    match http::content_type(content_type).as_str() {
        "application/x-www-form-urlencoded" | "" => {
            if body.is_empty() {
                return Ok(String::new());
            }
            let end = body.iter().rposition(|b| !b" \t\n\r\x0b\x0c".contains(b)).map_or(0, |i| i + 1);
            let text = http::decode(&body[..end], http::charset(content_type).as_deref()).map_err(|_| Reply::internal_error())?;
            Ok(http::parse_qsl(&text).into_iter().find(|(k, _)| k == "password").map(|(_, v)| v).unwrap_or_default())
        }
        "multipart/form-data" => {
            let boundary = multer::parse_boundary(content_type.unwrap_or("")).map_err(|_| Reply::internal_error())?;
            let stream = futures_util::stream::once(async move { Ok::<_, Infallible>(Bytes::from(body)) });
            let mut parts = multer::Multipart::new(stream, boundary);
            while let Some(field) = parts.next_field().await.map_err(|_| Reply::internal_error())? {
                if field.name() == Some("password") {
                    let text_like = field.content_type().is_none_or(|ct| ct.type_() == "text");
                    let value = field.bytes().await.map_err(|_| Reply::internal_error())?;
                    return Ok(if text_like { String::from_utf8(value.to_vec()).map_err(|_| Reply::internal_error())? } else { String::new() });
                }
            }
            Ok(String::new())
        }
        _ => Ok(String::new()),
    }
}

async fn serve(app: Arc<App>, listener: tokio::net::TcpListener) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                log!(app.ros.node.error(), "accept failed: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let app = Arc::clone(&app);
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |req| {
                let app = Arc::clone(&app);
                async move { Ok::<_, Infallible>(app.handle(req).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .title_case_headers(true)
                .serve_connection(TokioIo::new(stream), service)
                .with_upgrades()
                .await;
        });
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("web_server")?;
    let mut d = Declared { node: &node, handles: Vec::new() };
    let port: i64 = d.get("port", 8080)?;
    let map_frame: Arc<str> = d.get("map_frame", Arc::from("map"))?;
    let base_frame: Arc<str> = d.get("base_frame", Arc::from("camera_link"))?;
    let localization: bool = d.get("localization", false)?;
    let max_drive_speed: f64 = d.get("max_drive_speed", 0.2)?;
    let max_drive_turn: f64 = d.get("max_drive_turn", 1.0)?;
    let floor_grid: bool = d.get("floor_grid", true)?;
    let preview_fps: f64 = d.get("preview_fps", 4.0)?;
    let preview_topic: Arc<str> = d.get("preview_topic", Arc::from("/camera/left/image_rect_mono"))?;
    let preview_width: i64 = d.get("preview_width", 480)?;
    let depth_topic: Arc<str> = d.get("depth_topic", Arc::from("/camera/depth/image_rect_raw"))?;
    let depth_fps: f64 = d.get("depth_fps", 5.0)?;
    let depth_min_m: f64 = d.get("depth_min_m", 0.2)?;
    let depth_max_m: f64 = d.get("depth_max_m", 4.0)?;
    let depth_range_mm = ((depth_min_m * 1000.0).round().max(0.0) as u32, (depth_max_m * 1000.0).round().max(1.0) as u32);
    let port = u16::try_from(port).map_err(|_| anyhow!("port {port} out of range"))?;

    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let ros = Arc::new(Ros {
        tf: TfListener::new(&node)?,
        core: Mutex::new(Core::new(localization, max_drive_speed, max_drive_turn, floor_grid)),
        map_frame: map_frame.to_string(),
        base_frame: base_frame.to_string(),
        pub_zone_cmd: node.create_publisher("/perception/ignore_zones/command".keep_last(10))?,
        nav_client: node.create_action_client::<NavigateToPose>("navigate_to_pose")?,
        pub_cmd: node.create_publisher("/cmd_vel".keep_last(1))?,
        pub_cmd_nav: node.create_publisher("/cmd_vel_nav".keep_last(1))?,
        mode_client: node.create_client::<roomba_msgs::srv::SetMode>("/roomba/set_mode")?,
        pub_explore: node.create_publisher("/explorer/command".keep_last(10))?,
        pub_floor_grid: node.create_publisher("/perception/person_detector/floor_grid".qos(latched()))?,
        nav_requests: Mutex::new(VecDeque::new()),
        robot_requests: Mutex::new(VecDeque::new()),
        nav_handle: Mutex::new(None),
        drive_active: Mutex::new(false),
        jpeg: tokio::sync::watch::channel(None).0,
        last_annotated: Mutex::new(0.0),
        last_preview: Mutex::new(0.0),
        preview_sub: Mutex::new(None),
        preview_fps,
        preview_topic: preview_topic.to_string(),
        preview_width: preview_width.max(0) as usize,
        depth_jpeg: tokio::sync::watch::channel(None).0,
        depth_viewers: AtomicUsize::new(0),
        depth_sub: Mutex::new(None),
        last_depth: Mutex::new(0.0),
        depth_topic: depth_topic.to_string(),
        depth_fps: depth_fps.max(0.1),
        depth_colours: image::DepthColours::new(depth_range_mm.0 as f64, depth_range_mm.1 as f64),
        depth_range_mm,
        rt: rt.handle().clone(),
        node: node.clone(),
    });
    ros.pub_floor_grid.publish(std_msgs::msg::Bool { data: floor_grid }).ok();

    let mut keep: Vec<Box<dyn std::any::Any>> = Vec::new();
    macro_rules! sub {
        ($t:ty, $topic:expr, |$r:ident, $m:ident| $body:expr) => {{
            let $r = Arc::clone(&ros);
            keep.push(Box::new(node.create_subscription::<$t, _>($topic, move |$m: $t| $body)?));
        }};
    }
    sub!(sensor_msgs::msg::rmw::CompressedImage, "/perception/person_detector/annotated/compressed".keep_last(2), |r, m| r.on_jpeg(m));
    sub!(vision_msgs::msg::Detection3DArray, "/perception/person_detector/detections_3d".keep_last(5), |r, m| r.on_dets(m, false));
    sub!(vision_msgs::msg::Detection3DArray, "/perception/person_detector/ignored_3d".keep_last(5), |r, m| r.on_dets(m, true));
    // RTAB-Map publishes the grid latched and only on change; prefer the planner's (gap-filled) map.
    sub!(nav_msgs::msg::rmw::OccupancyGrid, "/rtabmap/map".qos(latched()), |r, m| r.on_map(m, false));
    sub!(nav_msgs::msg::rmw::OccupancyGrid, "/rtabmap/map_filled".qos(latched()), |r, m| r.on_map(m, true));
    sub!(nav_msgs::msg::rmw::Path, "/rtabmap/mapPath".keep_last(1), |r, m| lock(&r.core).on_path(&Ros::path_xy(&m)));
    sub!(std_msgs::msg::String, "/perception/ignore_zones".qos(latched()), |r, m| lock(&r.core).on_zones(&m.data));
    sub!(VisualSlamStatus, "/visual_slam/status".keep_last(10), |r, m| lock(&r.core).on_status(m.vo_state, mono()));
    sub!(geometry_msgs::msg::PoseWithCovarianceStamped, "/rtabmap/localization_pose".keep_last(1), |r, _m| lock(&r.core).on_localized());
    sub!(std_msgs::msg::String, "/nav_coach/instruction".keep_last(10), |r, m| lock(&r.core).on_instruction(&m.data, mono()));
    sub!(nav_msgs::msg::rmw::Path, "/plan".keep_last(1), |r, m| lock(&r.core).on_plan(&Ros::path_xy(&m)));
    sub!(nav_msgs::msg::rmw::OccupancyGrid, "/local_costmap/costmap".keep_last(1), |r, m| r.on_costmap(m));
    sub!(std_msgs::msg::String, "/perception/floor_status".keep_last(2), |r, m| lock(&r.core).on_floor(&m.data));
    sub!(std_msgs::msg::String, "/vslam_filter/status".keep_last(2), |r, m| lock(&r.core).on_filter(&m.data));
    {
        let r = Arc::clone(&ros);
        keep.push(Box::new(node.create_timer_repeating(Duration::from_millis(100).node_time(), move || r.process_nav_requests())?));
    }
    sub!(std_msgs::msg::String, "/roomba/mode".qos(latched()), |r, m| lock(&r.core).on_robot_mode(&m.data, mono()));
    sub!(sensor_msgs::msg::BatteryState, "/battery_state".keep_last(2), |r, m| r.on_battery(m));
    sub!(roomba_msgs::msg::SensorState, "/roomba/sensors".keep_last(2), |r, m| r.on_sensors(m));
    {
        let r = Arc::clone(&ros);
        keep.push(Box::new(node.create_timer_repeating(Duration::from_millis(100).node_time(), move || r.drive_tick())?));
    }
    sub!(std_msgs::msg::String, "/explorer/status".qos(latched()), |r, m| lock(&r.core).on_explore(&m.data, mono()));
    if preview_fps > 0.0 {
        // Plain camera preview while no annotated frames arrive (e.g. people:=false).
        let r = Arc::clone(&ros);
        keep.push(Box::new(node.create_timer_repeating(Duration::from_secs(2).node_time(), move || r.preview_check())?));
    }

    {
        let r = Arc::clone(&ros);
        keep.push(Box::new(node.create_timer_repeating(Duration::from_millis(500).node_time(), move || r.depth_check())?));
    }

    let www = http::package_share("roomba_web").ok_or_else(|| anyhow!("package 'roomba_web' not found"))?.join("www");
    let www = std::fs::canonicalize(&www).with_context(|| format!("{} missing", www.display()))?;
    let auth_paths = AuthPaths::default_paths();
    if auth::ensure_password(&auth_paths)? {
        log!(
            node.warn(),
            "No web password set; generated one in {} (change it with: ros2 run roomba_web set_password)",
            auth_paths.initial_file.display()
        );
    }

    let app = Arc::new(App {
        ros: Arc::clone(&ros),
        www,
        mime: MimeTypes::load(),
        auth_paths,
        sessions: Mutex::new(Sessions::default()),
        scrypt_slots: tokio::sync::Semaphore::new(10),
        streams: AtomicUsize::new(0),
    });
    // IPv6 and IPv4: [::] takes both where the system allows it, and then 0.0.0.0 can't bind (in
    // use); where it keeps IPv6 sockets IPv6-only, 0.0.0.0 binds too; without IPv6, only 0.0.0.0.
    let (v6, v4) = rt.block_on(async {
        (tokio::net::TcpListener::bind(("::", port)).await, tokio::net::TcpListener::bind(("0.0.0.0", port)).await)
    });
    let listeners: Vec<_> = match (v6, v4) {
        (Err(e), Err(_)) => return Err(anyhow!("cannot listen on port {port}: {e}")),
        (v6, v4) => {
            let what = match (&v6, &v4) {
                (Ok(_), Ok(_)) => "[::] (IPv6) and 0.0.0.0 (IPv4)".to_string(),
                (Ok(_), Err(e)) if e.kind() == std::io::ErrorKind::AddrInUse => "[::] (IPv6 and IPv4)".to_string(),
                (Ok(_), Err(e)) => format!("[::] (IPv6; not IPv4: {e})"),
                (Err(e), _) => format!("0.0.0.0 (IPv4 only: IPv6 {e})"),
            };
            log!(node.info(), "Web app on port {port}: listening on {what}");
            [v6.ok(), v4.ok()].into_iter().flatten().collect()
        }
    };
    std::thread::Builder::new().name("http".into()).spawn(move || {
        rt.block_on(async move {
            for l in listeners {
                tokio::spawn(serve(Arc::clone(&app), l));
            }
            std::future::pending::<()>().await
        })
    })?;
    // The executor also runs the drive pad deadman: if it stops, the whole node stops.
    roomba_rs::node_params::spin(&mut executor)?;
    drop(keep);
    Ok(())
}
