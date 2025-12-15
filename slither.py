import json
import math
import os
import re
import time
import io
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import cv2
import numpy as np
from PIL import Image

os.environ["DISPLAY"] = ":0"

# Set to True if running on Raspberry Pi
IS_RASPBERRY_PI = False


class SlitherController:
    def __init__(self, driver, save_screenshots=False, record_video=False):
        self.driver = driver
        self.radius = 50  # pixels from center
        self.current_angle = 0
        self.previous_angle = 0
        self.is_boosting = False
        self.save_screenshots = save_screenshots
        self.record_video = record_video
        self.frame_count = 0
        self.screenshot_folder = None
        self.images_folder = None
        self.game_start_time = datetime.now()

        # Video recording
        self.video_frames = [] if record_video else None
        self.video_annotations = (
            [] if record_video else None
        )  # Store annotations for each frame
        self.video_fps = (
            20  # Frames per second for video (increased to handle more frames)
        )

        # Game state log for JSON export
        self.game_log = {"game_start": self.game_start_time.isoformat(), "frames": []}

        if self.save_screenshots or self.record_video:
            self._setup_screenshot_folder()

    def _setup_screenshot_folder(self):
        """Create the games folder and timestamped subfolder for screenshots."""
        games_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
        os.makedirs(games_folder, exist_ok=True)

        # Create subfolder with game start timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.screenshot_folder = os.path.join(games_folder, timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)

        # Create images subfolder for screenshots (only if saving screenshots)
        if self.save_screenshots:
            self.images_folder = os.path.join(self.screenshot_folder, "images")
            os.makedirs(self.images_folder, exist_ok=True)
            print(f"Screenshots will be saved to: {self.images_folder}")
        else:
            self.images_folder = None

        if self.record_video:
            print(f"Video will be saved to: {self.screenshot_folder}")

    def capture_screenshot(self, action_type="move"):
        """Capture a screenshot with metadata and log game state."""
        # Capture if saving screenshots OR recording video
        if (
            not (self.save_screenshots or self.record_video)
            or not self.screenshot_folder
        ):
            return

        boost_str = "boost" if self.is_boosting else "normal"
        filename = f"frame_{self.frame_count:06d}_angle_{self.current_angle:.1f}_speed_{boost_str}.png"
        filepath = (
            os.path.join(self.images_folder, filename) if self.images_folder else None
        )

        # IMPORTANT: Capture game state FIRST, then screenshot immediately after
        # This minimizes timing mismatch between state and image
        detailed_state = self.get_detailed_state()
        snake_length = self.get_snake_length()
        snake_rank = self.get_snake_rank()

        # Take screenshot IMMEDIATELY after getting state
        screenshot_png = self.driver.get_screenshot_as_png()

        if self.save_screenshots and filepath:
            # Save screenshot to file
            with open(filepath, "wb") as f:
                f.write(screenshot_png)

        # Store frame for video if recording
        if self.record_video:
            # Convert screenshot to numpy array for video
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            # Store annotation placeholder (will be set by set_frame_annotation if needed)
            self.video_annotations.append(None)

        # Now record timestamp (after capture)
        current_time = datetime.now()
        elapsed_time = (current_time - self.game_start_time).total_seconds()

        # Create frame state entry
        frame_state = {
            "frame_id": self.frame_count,
            "timestamp": current_time.isoformat(),
            "elapsed_seconds": round(elapsed_time, 3),
            "image_file": f"images/{filename}" if self.save_screenshots else None,
            "state": {
                "current_angle": self.current_angle,
                "previous_angle": self.previous_angle,
                "angle_change": self.current_angle - self.previous_angle,
                "is_boosting": self.is_boosting,
                "snake_length": snake_length,
                "snake_rank": snake_rank,
            },
            "action": {
                "type": action_type,
                "target_angle": self.current_angle,
                "boost_active": self.is_boosting,
            },
            "game_world": detailed_state,
        }

        # Add previous frames context (last 5 actions) - only if saving screenshots
        if self.save_screenshots:
            if len(self.game_log["frames"]) > 0:
                recent_frames = self.game_log["frames"][-5:]
                frame_state["previous_actions"] = [
                    {
                        "frame_id": f["frame_id"],
                        "angle": f["state"]["current_angle"],
                        "boosting": f["state"]["is_boosting"],
                        "action_type": f["action"]["type"],
                        "length": f["state"].get("snake_length", 0),
                    }
                    for f in recent_frames
                ]
            else:
                frame_state["previous_actions"] = []

            # Add to game log
            self.game_log["frames"].append(frame_state)

        self.frame_count += 1

        # Auto-save JSON every 10 frames (only if saving screenshots)
        if self.save_screenshots and self.frame_count % 10 == 0:
            self.save_game_log()

    def save_game_log(self):
        """Save the game log to a JSON file."""
        if not self.screenshot_folder:
            return

        # Update game end time
        self.game_log["game_end"] = datetime.now().isoformat()
        self.game_log["total_frames"] = self.frame_count
        self.game_log["duration_seconds"] = (
            datetime.now() - self.game_start_time
        ).total_seconds()

        # Add final stats
        self.game_log["final_length"] = self.get_snake_length()
        self.game_log["final_rank"] = self.get_snake_rank()

        # Calculate max length achieved during the game
        if self.game_log["frames"]:
            lengths = [
                f["state"].get("snake_length", 0) for f in self.game_log["frames"]
            ]
            self.game_log["max_length"] = max(lengths) if lengths else 0

        json_path = None
        if self.save_screenshots:
            json_path = os.path.join(self.screenshot_folder, "game_log.json")
            with open(json_path, "w") as f:
                json.dump(self.game_log, f, indent=2)

        # Create video if recording was enabled
        if self.record_video and self.video_frames:
            video_path = self._create_video()
            if video_path:
                print(f"Video saved to: {video_path}")

        return json_path

    def _create_video(self):
        """Create a video from collected frames with annotations."""
        if not self.video_frames:
            print("Warning: No video frames to save")
            return None

        if not self.screenshot_folder:
            print("Warning: No screenshot folder set for video saving")
            return None

        if len(self.video_frames) == 0:
            print("Warning: Video frames list is empty")
            return None

        try:
            # Get frame dimensions from first frame
            height, width = self.video_frames[0].shape[:2]
            print(
                f"Creating video with {len(self.video_frames)} frames at {self.video_fps} FPS, resolution: {width}x{height}"
            )

            # Define video codec and create VideoWriter
            video_filename = os.path.join(self.screenshot_folder, "gameplay.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                video_filename, fourcc, self.video_fps, (width, height)
            )

            if not out.isOpened():
                print(f"Error: Failed to open video writer for {video_filename}")
                return None

            # Write all frames with annotations
            for i, frame in enumerate(self.video_frames):
                annotated_frame = self._annotate_frame(frame, i)
                out.write(annotated_frame)
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(self.video_frames)} frames...")

            # Release everything
            out.release()

            # Verify file was created
            if os.path.exists(video_filename):
                file_size = os.path.getsize(video_filename)
                print(
                    f"Video successfully created: {video_filename} ({file_size / 1024 / 1024:.2f} MB)"
                )
            else:
                print(f"Warning: Video file was not created at {video_filename}")
                return None

            # Clear frames from memory
            self.video_frames = []
            self.video_annotations = []

            return video_filename
        except Exception as e:
            import traceback

            print(f"Error creating video: {e}")
            traceback.print_exc()
            return None

    def _annotate_frame(self, frame, frame_idx):
        """Add annotations (observation and logits) to a frame."""
        if not self.video_annotations or frame_idx >= len(self.video_annotations):
            return frame

        annotation = self.video_annotations[frame_idx]
        # If no annotation for this frame, try to use the previous frame's annotation
        # (for intermediate frames captured during delays)
        if not annotation:
            # Look backwards for the most recent annotation
            for i in range(frame_idx - 1, -1, -1):
                if i < len(self.video_annotations) and self.video_annotations[i]:
                    annotation = self.video_annotations[i]
                    break
            if not annotation:
                return frame

        # Create a copy of the frame to annotate
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_height = 20
        x_offset = 10
        y_start = 30

        y_pos = y_start

        # Draw background rectangle for better readability
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (x_offset - 5, y_start - 20),
            (width - 10, y_start + 300),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # Title
        cv2.putText(
            annotated_frame,
            "Model State:",
            (x_offset, y_pos),
            font,
            font_scale + 0.2,
            (0, 255, 255),
            thickness + 1,
        )
        y_pos += line_height + 5

        # Observation values
        if annotation.get("observation") is not None:
            obs = annotation["observation"]
            if isinstance(obs, np.ndarray):
                obs = obs.tolist()
            obs_labels = [
                "Angle",
                "Length",
                "Food Dist",
                "Food Ang",
                "Prey Dist",
                "Prey Ang",
                "Enemy Dist",
                "Enemy Ang",
                "Food Cnt",
                "Prey Cnt",
                "Enemy Cnt",
                "Food Q1",
                "Food Q2",
                "Food Q3",
                "Food Q4",
            ]
            cv2.putText(
                annotated_frame,
                "Observation:",
                (x_offset, y_pos),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
            )
            y_pos += line_height

            for i, (label, value) in enumerate(zip(obs_labels, obs)):
                if i < len(obs):
                    text = f"  {label}: {value:.3f}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x_offset, y_pos),
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    y_pos += line_height - 2
                    if y_pos > height - 100:  # Prevent overflow
                        break

        y_pos += 5

        # Action probabilities/logits
        if annotation.get("probabilities") is not None:
            probs = annotation["probabilities"]
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            action = annotation.get("action", -1)
            action_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]

            cv2.putText(
                annotated_frame,
                "Action Probabilities (Degrees):",
                (x_offset, y_pos),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
            )
            y_pos += line_height

            for i, (label, prob) in enumerate(zip(action_labels, probs)):
                if i < len(probs):
                    # Highlight selected action
                    text_color = (0, 255, 0) if i == action else color
                    text = f"  {label}: {prob:.3f}"
                    if i == action:
                        text += " <--"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x_offset, y_pos),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                    )
                    y_pos += line_height - 2

        elif annotation.get("logits") is not None:
            logits = annotation["logits"]
            if isinstance(logits, np.ndarray):
                logits = logits.tolist()
            action = annotation.get("action", -1)
            action_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]

            cv2.putText(
                annotated_frame,
                "Action Logits:",
                (x_offset, y_pos),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
            )
            y_pos += line_height

            for i, (label, logit) in enumerate(zip(action_labels, logits)):
                if i < len(logits):
                    # Highlight selected action
                    text_color = (0, 255, 0) if i == action else color
                    text = f"  {label}: {logit:.3f}"
                    if i == action:
                        text += " <--"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x_offset, y_pos),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                    )
                    y_pos += line_height - 2

        return annotated_frame

    def set_frame_annotation(
        self, observation=None, logits=None, probabilities=None, action=None
    ):
        """Set annotation for the most recently captured frame."""
        if (
            self.record_video
            and self.video_annotations
            and len(self.video_annotations) > 0
        ):
            self.video_annotations[-1] = {
                "observation": observation,
                "logits": logits,
                "probabilities": probabilities,
                "action": action,
            }

    def get_snake_length(self):
        """Get the current snake length by extracting from page text."""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            # Match "Your length: [number]" (case insensitive)
            match = re.search(r"Your length:\s*(\d+)", page_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        except Exception:
            return 0

    def get_snake_rank(self):
        """Get the current snake rank by extracting from page text."""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            # Match "Your rank: [X] of [Y]" (case insensitive)
            match = re.search(
                r"Your rank:\s*(\d+)\s*of\s*(\d+)", page_text, re.IGNORECASE
            )
            if match:
                return {
                    "rank": int(match.group(1)),
                    "total_players": int(match.group(2)),
                }
            return {"rank": 0, "total_players": 0}
        except Exception:
            return {"rank": 0, "total_players": 0}

    def get_detailed_state(self):
        """
        Get detailed game state including snake position, nearby foods, and other snakes.
        Uses discovered variable names: slithers (snakes), foods (food pellets).

        Enemy snake detection: Checks entire body (head + all segments) and returns
        distance to the closest body part for accurate collision avoidance.

        Returns structured data useful for RL training.
        """
        try:
            state = self.driver.execute_script(
                """
                var result = {
                    snake: null,
                    foods: [],
                    other_snakes: [],
                    world: {
                        scale: window.gsc || 1,
                        boundary: window.grd || 32550
                    }
                };
                
                // Find our snake - try multiple approaches
                var mySnake = null;
                
                // Method 1: Try window.snake directly
                if (window.snake && window.snake.xx !== undefined) {
                    mySnake = window.snake;
                }
                
                // Method 2: Look for 'playing' or 'me' property in slithers
                if (!mySnake && window.slithers) {
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && (s.me === true || s.playing === true || s.isMine === true)) {
                            mySnake = s;
                            break;
                        }
                    }
                }
                
                // Method 3: Use view center position (view_xx, view_yy)
                // Our snake should be at the view center
                if (!mySnake && window.view_xx !== undefined && window.slithers) {
                    var viewX = window.view_xx;
                    var viewY = window.view_yy;
                    var minDist = Infinity;
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && s.xx !== undefined) {
                            var dx = s.xx - viewX;
                            var dy = s.yy - viewY;
                            var dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < minDist) {
                                minDist = dist;
                                mySnake = s;
                            }
                        }
                    }
                }
                
                // Method 4: Fallback - first slither (might be wrong!)
                if (!mySnake && window.slithers && window.slithers.length > 0) {
                    mySnake = window.slithers[0];
                }
                
                if (!mySnake || mySnake.xx === undefined) {
                    return result;  // Game not fully loaded
                }
                
                var sx = mySnake.xx;
                var sy = mySnake.yy;
                
                result.snake = {
                    x: sx,
                    y: sy,
                    id: mySnake.id,
                    angle: mySnake.ang || mySnake.ehang || 0,
                    speed: mySnake.sp || 0,
                    length: mySnake.sct || 0,
                    // Debug: how was this snake found?
                    found_via: mySnake === window.snake ? 'window.snake' : 'slithers_search'
                };
                
                // Collect foods - using discovered variable name 'foods'
                // Keys: id, xx, yy, rx, ry, rsp, cv, rad, sz, lrrad
                if (window.foods && window.foods.length > 0) {
                    var foodList = [];
                    for (var i = 0; i < window.foods.length; i++) {
                        var f = window.foods[i];
                        if (f && f.xx !== undefined && f.yy !== undefined) {
                            var dx = f.xx - sx;
                            var dy = f.yy - sy;
                            var dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < 1500) {  // Only foods within range
                                foodList.push({
                                    id: f.id,
                                    x: f.xx,
                                    y: f.yy,
                                    size: f.sz || 1,
                                    radius: f.rad || 1,
                                    distance: Math.round(dist),
                                    angle: Math.atan2(dy, dx)
                                });
                            }
                        }
                    }
                    foodList.sort(function(a, b) { return a.distance - b.distance; });
                    result.foods = foodList.slice(0, 50);  // Keep closest 50
                }
                
                // Collect other snakes - using discovered variable name 'slithers'
                // Keys: id, xx, yy, rcv, er, pr, pma, ec, ecv, eca
                // Now checking entire body, not just head
                if (window.slithers && window.slithers.length > 0) {
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && s.xx !== undefined && s !== mySnake && s.id !== mySnake.id) {
                            // Head position
                            var head_dx = s.xx - sx;
                            var head_dy = s.yy - sy;
                            var head_dist = Math.sqrt(head_dx*head_dx + head_dy*head_dy);
                            
                            // Find closest point on snake body
                            var min_body_dist = head_dist;
                            var closest_x = s.xx;
                            var closest_y = s.yy;
                            
                            // Check body segments if available
                            // Body points are stored in pts array with indices alternating x,y
                            if (s.pts && s.pts.length > 0) {
                                for (var j = 0; j < s.pts.length; j++) {
                                    var pt = s.pts[j];
                                    if (pt && pt.xx !== undefined && pt.yy !== undefined) {
                                        var bdx = pt.xx - sx;
                                        var bdy = pt.yy - sy;
                                        var bdist = Math.sqrt(bdx*bdx + bdy*bdy);
                                        if (bdist < min_body_dist) {
                                            min_body_dist = bdist;
                                            closest_x = pt.xx;
                                            closest_y = pt.yy;
                                        }
                                    }
                                }
                            }
                            
                            if (min_body_dist < 3000) {  // Only snakes within range
                                // Collect all body segment positions
                                var body_segments = [];
                                if (s.pts && s.pts.length > 0) {
                                    for (var k = 0; k < s.pts.length; k++) {
                                        var seg = s.pts[k];
                                        if (seg && seg.xx !== undefined && seg.yy !== undefined) {
                                            body_segments.push({
                                                x: seg.xx,
                                                y: seg.yy
                                            });
                                        }
                                    }
                                }
                                
                                result.other_snakes.push({
                                    id: s.id,
                                    head_x: s.xx,
                                    head_y: s.yy,
                                    closest_x: closest_x,
                                    closest_y: closest_y,
                                    angle: s.ang || s.ehang || 0,
                                    speed: s.sp || 0,
                                    length: s.sct || 0,
                                    distance: Math.round(min_body_dist),  // Distance to closest body part
                                    head_distance: Math.round(head_dist),  // Distance to head
                                    angle_to: Math.atan2(closest_y - sy, closest_x - sx),  // Angle to closest part
                                    body_segments: body_segments  // All body segment positions
                                });
                            }
                        }
                    }
                    result.other_snakes.sort(function(a, b) { return a.distance - b.distance; });
                    result.other_snakes = result.other_snakes.slice(0, 15);  // Keep closest 15
                }
                
                // Also check preys (special food items that appear when snakes die)
                if (window.preys && window.preys.length > 0) {
                    result.preys = [];
                    for (var i = 0; i < window.preys.length; i++) {
                        var p = window.preys[i];
                        if (p && p.xx !== undefined) {
                            var dx = p.xx - sx;
                            var dy = p.yy - sy;
                            var dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < 2000) {
                                result.preys.push({
                                    x: p.xx,
                                    y: p.yy,
                                    size: p.sz || 1,
                                    distance: Math.round(dist),
                                    angle: Math.atan2(dy, dx)
                                });
                            }
                        }
                    }
                    result.preys.sort(function(a, b) { return a.distance - b.distance; });
                    result.preys = result.preys.slice(0, 20);
                }
                
                return result;
                """
            )
            return state
        except Exception as e:
            print(f"Error getting detailed state: {e}")
            return None

    def move_to_angle(self, angle_degrees):
        """
        Move cursor to a position at the specified angle from center.
        Uses standard mathematical angles: 0° = right, 90° = up, 180° = left, 270° = down
        """
        self.previous_angle = self.current_angle
        self.current_angle = angle_degrees % 360
        angle_radians = math.radians(angle_degrees)

        # Calculate offset from center
        # Negate Y because screen coordinates have Y increasing downward
        offset_x = int(self.radius * math.cos(angle_radians))
        offset_y = int(-self.radius * math.sin(angle_radians))

        # Slither.io stores mouse coordinates and snake angle in global variables
        self.driver.execute_script(
            f"""
            // Set the target angle directly on the snake object (slither uses radians)
            if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                window.snake.ang = {angle_radians};
                window.snake.wang = {angle_radians};
                window.snake.eang = {angle_radians};
            }}
            
            // Set mouse position variables used by the game
            window.xm = {offset_x};
            window.ym = {offset_y};
            
            // Some versions use these global mouse coordinates
            window.mouseX = window.innerWidth / 2 + {offset_x};
            window.mouseY = window.innerHeight / 2 + {offset_y};
            
            // Dispatch mousemove event on document
            var event = new MouseEvent('mousemove', {{
                view: window,
                bubbles: true,
                cancelable: true,
                clientX: window.innerWidth / 2 + {offset_x},
                clientY: window.innerHeight / 2 + {offset_y}
            }});
            document.dispatchEvent(event);
        """
        )

        # Capture screenshot after movement
        self.capture_screenshot(action_type="move")

    def boost(self, enabled=True):
        """Hold mouse button to boost (speed up) while maintaining current direction."""
        self.is_boosting = enabled

        # Calculate offset for current angle to maintain direction
        angle_radians = math.radians(self.current_angle)
        offset_x = int(self.radius * math.cos(angle_radians))
        offset_y = int(-self.radius * math.sin(angle_radians))

        if enabled:
            self.driver.execute_script(
                f"""
                // Maintain current angle while boosting
                if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                    window.snake.ang = {angle_radians};
                    window.snake.wang = {angle_radians};
                    window.snake.eang = {angle_radians};
                }}
                window.xm = {offset_x};
                window.ym = {offset_y};
                
                // Trigger boost (mousedown)
                var canvas = document.querySelector('canvas');
                var event = new MouseEvent('mousedown', {{ bubbles: true }});
                canvas.dispatchEvent(event);
            """
            )
        else:
            self.driver.execute_script(
                f"""
                // Maintain current angle when stopping boost
                if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                    window.snake.ang = {angle_radians};
                    window.snake.wang = {angle_radians};
                    window.snake.eang = {angle_radians};
                }}
                window.xm = {offset_x};
                window.ym = {offset_y};
                
                // Stop boost (mouseup)
                var canvas = document.querySelector('canvas');
                var event = new MouseEvent('mouseup', {{ bubbles: true }});
                canvas.dispatchEvent(event);
            """
            )

        # Capture screenshot after boost state change
        action_type = "boost_start" if enabled else "boost_end"
        self.capture_screenshot(action_type=action_type)

    def maintain_direction(self, duration=1.0, interval=0.05):
        """
        Maintain current direction for a specified duration.
        Continuously updates the angle to prevent drift.
        Use this instead of time.sleep() when you want to hold a direction.
        """
        steps = int(duration / interval)
        for _ in range(steps):
            # Re-apply the current angle
            angle_radians = math.radians(self.current_angle)
            offset_x = int(self.radius * math.cos(angle_radians))
            offset_y = int(-self.radius * math.sin(angle_radians))

            self.driver.execute_script(
                f"""
                if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                    window.snake.ang = {angle_radians};
                    window.snake.wang = {angle_radians};
                    window.snake.eang = {angle_radians};
                }}
                window.xm = {offset_x};
                window.ym = {offset_y};
            """
            )
            # Capture frame periodically if recording video
            if (
                self.record_video and self.frame_count % 2 == 0
            ):  # Capture every other step
                self.capture_screenshot(action_type="hold")
            time.sleep(interval)

        # Take a screenshot at the end
        self.capture_screenshot(action_type="hold")

    def capture_frame_only(self):
        """Capture a frame for video without saving screenshot or logging (for frequent captures)."""
        if not self.record_video or not self.screenshot_folder:
            return

        try:
            screenshot_png = self.driver.get_screenshot_as_png()
            # Convert screenshot to numpy array for video
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            # Store annotation placeholder (will be set by set_frame_annotation if needed)
            self.video_annotations.append(None)
            self.frame_count += 1
        except Exception as e:
            print(f"Error capturing frame: {e}")

    def is_game_over(self):
        """Check if the game over screen is displayed."""
        try:
            # Look for the play button which appears on game over
            play_buttons = self.driver.find_elements(By.CLASS_NAME, "btnt")
            for button in play_buttons:
                if "Play" in button.text and button.is_displayed():
                    return True
            return False
        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    # Setup driver
    if IS_RASPBERRY_PI:
        # Raspberry Pi configuration with kiosk mode
        options = Options()
        options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument("--kiosk")  # Fullscreen with no navigation bar
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        service = Service("/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Standard Chrome driver
        driver = webdriver.Chrome()

    driver.get("http://slither.io")

    # Wait for game to load and start
    time.sleep(5)  # Adjust as needed, or use explicit waits

    # Set game to low quality for better performance
    try:
        driver.find_element(By.ID, "grqi").click()
        print("Set game to low quality")
    except Exception:
        pass  # Quality button may not be available

    # start the game
    play_button = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_button:
        if "Play" in button.text:
            button.click()
            break
    print("Game started")
    time.sleep(1)

    # Create controller with screenshot saving and video recording enabled
    controller = SlitherController(driver, save_screenshots=True, record_video=False)

    # Food-seeking policy with danger avoidance
    print("Starting food-seeking policy with danger avoidance...")

    DANGER_DISTANCE = 300  # Flee if enemy snake is closer than this
    SAFE_DISTANCE = 500  # Stop fleeing once enemy is farther than this

    fleeing = False

    while True:
        # Check for game over
        if controller.is_game_over():
            print("Game over detected!")
            break

        # Get current game state
        state = controller.get_detailed_state()

        if not state:
            controller.maintain_direction(0.5)
            continue

        # Check for nearby enemy snakes
        other_snakes = state.get("other_snakes", [])
        nearest_enemy = other_snakes[0] if other_snakes else None

        # Decide: flee or seek food
        if nearest_enemy and nearest_enemy["distance"] < DANGER_DISTANCE:
            # DANGER! Flee from the nearest enemy
            fleeing = True

            # Calculate angle AWAY from enemy (opposite direction)
            angle_to_enemy_rad = nearest_enemy["angle_to"]
            angle_away_deg = (
                math.degrees(angle_to_enemy_rad) + 180
            )  # Opposite direction

            # Convert to our coordinate system
            target_angle = -angle_away_deg % 360

            controller.move_to_angle(target_angle)
            time.sleep(0.05)  # Faster updates when fleeing

        elif fleeing and nearest_enemy and nearest_enemy["distance"] < SAFE_DISTANCE:
            # Still fleeing, not safe yet
            angle_to_enemy_rad = nearest_enemy["angle_to"]
            angle_away_deg = math.degrees(angle_to_enemy_rad) + 180
            target_angle = -angle_away_deg % 360

            controller.move_to_angle(target_angle)
            time.sleep(0.05)

        else:
            # Safe - seek food
            fleeing = False

            foods = state.get("foods", [])
            if foods:
                # Find nearest food
                nearest_food = foods[0]  # Already sorted by distance

                # Calculate angle to food
                angle_to_food_rad = nearest_food["angle"]
                angle_to_food_deg = math.degrees(angle_to_food_rad)

                # Convert to our coordinate system
                target_angle = -angle_to_food_deg % 360

                controller.move_to_angle(target_angle)
                time.sleep(0.1)
            else:
                # No food visible, maintain current direction
                controller.maintain_direction(0.5)
    # Save final game log
    log_path = controller.save_game_log()
    if controller.save_screenshots:
        print(f"Game session complete. {controller.frame_count} screenshots saved.")
        if log_path:
            print(f"Game log saved to: {log_path}")
    if controller.record_video:
        print(f"Video recording complete. {controller.frame_count} frames captured.")
    driver.quit()
