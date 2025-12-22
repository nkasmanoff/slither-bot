"""Slither.io game controller using Selenium WebDriver.

Provides the SlitherController class for interacting with the game,
including movement, state extraction, and video recording.
"""

import io
import json
import math
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from selenium.webdriver.common.by import By


class SlitherController:
    """Controller for interacting with the Slither.io game via Selenium."""

    def __init__(self, driver, save_screenshots=False, record_video=False):
        self.driver = driver
        self.radius = 50
        self.current_angle = 0
        self.previous_angle = 0
        self.is_boosting = False
        self.save_screenshots = save_screenshots
        self.record_video = record_video
        self.frame_count = 0
        self.screenshot_folder = None
        self.images_folder = None
        self.game_start_time = datetime.now()
        self.video_frames = [] if record_video else None
        self.video_annotations = [] if record_video else None
        self.video_fps = 20
        self.game_log = {"game_start": self.game_start_time.isoformat(), "frames": []}

        if self.save_screenshots or self.record_video:
            self._setup_screenshot_folder()

    def _setup_screenshot_folder(self):
        """Create folders for screenshots and video output."""
        games_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "games"
        )
        os.makedirs(games_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.screenshot_folder = os.path.join(games_folder, timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)

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

        detailed_state = self.get_detailed_state()
        snake_length = self.get_snake_length()
        snake_rank = self.get_snake_rank()
        screenshot_png = self.driver.get_screenshot_as_png()

        if self.save_screenshots and filepath:
            with open(filepath, "wb") as f:
                f.write(screenshot_png)

        if self.record_video:
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            self.video_annotations.append(None)

        current_time = datetime.now()
        elapsed_time = (current_time - self.game_start_time).total_seconds()

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

            self.game_log["frames"].append(frame_state)

        self.frame_count += 1

        if self.save_screenshots and self.frame_count % 10 == 0:
            self.save_game_log()

    def save_game_log(self):
        """Save the game log to a JSON file."""
        if not self.screenshot_folder:
            return

        self.game_log["game_end"] = datetime.now().isoformat()
        self.game_log["total_frames"] = self.frame_count
        self.game_log["duration_seconds"] = (
            datetime.now() - self.game_start_time
        ).total_seconds()
        self.game_log["final_length"] = self.get_snake_length()
        self.game_log["final_rank"] = self.get_snake_rank()

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

        if self.record_video and self.video_frames:
            video_path = self._create_video()
            if video_path:
                print(f"Video saved to: {video_path}")

        return json_path

    def _create_video(self):
        """Create screenshots and a GIF from collected frames with annotations."""
        if not self.video_frames or not self.screenshot_folder:
            return None

        try:
            print(f"Processing {len(self.video_frames)} frames...")

            # Create screenshots folder
            screenshots_folder = os.path.join(self.screenshot_folder, "screenshots")
            os.makedirs(screenshots_folder, exist_ok=True)

            # Process and save each frame as PNG, collect for GIF
            pil_images = []
            for i, frame in enumerate(self.video_frames):
                annotated_frame = self._annotate_frame(frame, i)

                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Save screenshot
                screenshot_path = os.path.join(screenshots_folder, f"frame_{i:04d}.png")
                pil_img.save(screenshot_path)

                # Collect for GIF
                pil_images.append(pil_img)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(self.video_frames)} frames...")

            print(f"Screenshots saved to: {screenshots_folder}")

            # Create GIF with 0.5 second delay (500ms)
            gif_filename = os.path.join(self.screenshot_folder, "gameplay.gif")
            if pil_images:
                pil_images[0].save(
                    gif_filename,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=500,  # 0.5 seconds between frames
                    loop=0,  # Loop forever
                )

                if os.path.exists(gif_filename):
                    file_size = os.path.getsize(gif_filename)
                    print(
                        f"GIF created: {gif_filename} ({file_size / 1024 / 1024:.2f} MB)"
                    )

            self.video_frames = []
            self.video_annotations = []
            return gif_filename

        except Exception as e:
            print(f"Error creating screenshots/GIF: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _annotate_frame(self, frame, frame_idx):
        """Add annotations to a video frame with discrete action display."""
        if not self.video_annotations or frame_idx >= len(self.video_annotations):
            return frame

        annotation = self.video_annotations[frame_idx]
        if not annotation:
            for i in range(frame_idx - 1, -1, -1):
                if i < len(self.video_annotations) and self.video_annotations[i]:
                    annotation = self.video_annotations[i]
                    break
            if not annotation:
                return frame

        annotated_frame = frame.copy()
        height, width = frame.shape[:2]

        # Constants for discrete actions
        NUM_ACTIONS = 12
        ANGLE_PER_ACTION = 360.0 / NUM_ACTIONS  # 30 degrees per action

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        line_height = 16
        x_offset = 10
        y_start = 25
        y_pos = y_start

        # Observation feature labels (23 features)
        obs_labels = [
            "current_angle",
            "snake_length",
            "nearest_food_dist",
            "food_action",
            "nearest_prey_dist",
            "prey_action",
            "nearest_enemy_dist",
            "enemy_action",
            "nearest_enemy_head",
            "num_foods",
            "num_preys",
            "num_enemies",
            "food_efficiency",
            "enemy_threat",
            "danger_front",
            "danger_right",
            "danger_back",
            "danger_left",
            "enemy_in_front",
            "enemy_in_right",
            "enemy_in_left",
            "last_action_sin",
            "last_action_cos",
        ]

        # Calculate overlay height based on content
        obs = annotation.get("observation")
        num_obs_lines = len(obs_labels) if obs is not None else 0
        overlay_height = 80 + num_obs_lines * (line_height - 2)

        # Semi-transparent overlay for text background
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (x_offset - 5, y_start - 15),
            (220, min(y_start + overlay_height, height - 10)),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        cv2.putText(
            annotated_frame,
            "Model State:",
            (x_offset, y_pos),
            font,
            font_scale + 0.1,
            (0, 255, 255),
            thickness,
        )
        y_pos += line_height + 2

        # Get action and compute angle
        action_idx = annotation.get("action")
        action_angle = None
        if action_idx is not None:
            action_angle = action_idx * ANGLE_PER_ACTION
            cv2.putText(
                annotated_frame,
                f"Action: {action_idx} ({action_angle:.0f} deg)",
                (x_offset, y_pos),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
            )
            y_pos += line_height

        # Show all observation values
        if obs is not None:
            if isinstance(obs, np.ndarray):
                obs = obs.tolist()

            cv2.putText(
                annotated_frame,
                "Observations:",
                (x_offset, y_pos),
                font,
                font_scale,
                (255, 200, 100),
                thickness,
            )
            y_pos += line_height

            for i, (label, value) in enumerate(zip(obs_labels, obs)):
                if i < len(obs) and y_pos < height - 20:
                    # Color code: green for positive, red for negative, white for zero
                    if value > 0.5:
                        val_color = (100, 255, 100)  # Green
                    elif value < -0.1:
                        val_color = (100, 100, 255)  # Red (BGR)
                    else:
                        val_color = (200, 200, 200)  # Gray

                    text = f"{label}: {value:.2f}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x_offset, y_pos),
                        font,
                        font_scale - 0.05,
                        val_color,
                        thickness,
                    )
                    y_pos += line_height - 2

        # Draw direction arrow at center of screen
        if action_angle is not None:
            center_x = width // 2
            center_y = height // 2
            arrow_length = 60
            arrow_head_length = 15

            # Convert angle to radians (0 degrees = right, counter-clockwise)
            angle_rad = math.radians(action_angle)
            end_x = int(center_x + arrow_length * math.cos(angle_rad))
            end_y = int(
                center_y - arrow_length * math.sin(angle_rad)
            )  # Negative because y increases downward

            # Draw arrow line
            cv2.arrowedLine(
                annotated_frame,
                (center_x, center_y),
                (end_x, end_y),
                (0, 255, 0),  # Green arrow
                3,  # Thickness
                tipLength=0.3,
            )

            # Draw center circle
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

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
        """Get the current snake length from the UI."""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            match = re.search(r"Your length:\s*(\d+)", page_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        except Exception:
            return 0

    def get_snake_rank(self):
        """Get the current snake rank from the UI."""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
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
        """Get detailed game state including snake position, foods, and other snakes."""
        result = self.get_full_state()
        if result is None:
            return None
        return {
            "snake": result.get("snake"),
            "foods": result.get("foods", []),
            "other_snakes": result.get("other_snakes", []),
            "preys": result.get("preys", []),
            "world": result.get("world", {}),
        }

    def get_full_state(self):
        """Get all game state in a single JS call for efficiency."""
        try:
            result = self.driver.execute_script(
                """
                var result = {
                    snake: null, foods: [], other_snakes: [], preys: [],
                    world: { scale: window.gsc || 1, boundary: window.grd || 32550 },
                    snake_length: 0, is_game_over: false, rank: {rank: 0, total_players: 0}
                };
                
                var deathOverlay = document.querySelector('.nsi');
                if (deathOverlay && deathOverlay.style.display !== 'none') result.is_game_over = true;
                
                var playBtns = document.querySelectorAll('.btnt');
                for (var i = 0; i < playBtns.length; i++) {
                    if (playBtns[i].textContent.indexOf('Play') >= 0 && playBtns[i].offsetParent !== null) {
                        result.is_game_over = true; break;
                    }
                }
                
                var bodyText = document.body.innerText || '';
                var lengthMatch = bodyText.match(/Your length:\\s*(\\d+)/i);
                if (lengthMatch) result.snake_length = parseInt(lengthMatch[1], 10);
                
                var rankMatch = bodyText.match(/Your rank:\\s*(\\d+)\\s*of\\s*(\\d+)/i);
                if (rankMatch) result.rank = { rank: parseInt(rankMatch[1], 10), total_players: parseInt(rankMatch[2], 10) };
                
                var mySnake = null;
                if (window.snake && window.snake.xx !== undefined) mySnake = window.snake;
                
                if (!mySnake && window.slithers) {
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && (s.me === true || s.playing === true || s.isMine === true)) { mySnake = s; break; }
                    }
                }
                
                if (!mySnake && window.view_xx !== undefined && window.slithers) {
                    var viewX = window.view_xx, viewY = window.view_yy, minDist = Infinity;
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && s.xx !== undefined) {
                            var dx = s.xx - viewX, dy = s.yy - viewY, dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < minDist) { minDist = dist; mySnake = s; }
                        }
                    }
                }
                
                if (!mySnake && window.slithers && window.slithers.length > 0) mySnake = window.slithers[0];
                if (!mySnake || mySnake.xx === undefined) return result;
                
                var sx = mySnake.xx, sy = mySnake.yy;
                result.snake = {
                    x: sx, y: sy, id: mySnake.id, angle: mySnake.ang || mySnake.ehang || 0,
                    speed: mySnake.sp || 0, length: mySnake.sct || 0,
                    found_via: mySnake === window.snake ? 'window.snake' : 'slithers_search'
                };
                
                if (window.foods && window.foods.length > 0) {
                    var foodList = [];
                    for (var i = 0; i < window.foods.length; i++) {
                        var f = window.foods[i];
                        if (f && f.xx !== undefined && f.yy !== undefined) {
                            var dx = f.xx - sx, dy = f.yy - sy, dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < 1500) {
                                foodList.push({ id: f.id, x: f.xx, y: f.yy, size: f.sz || 1, radius: f.rad || 1,
                                    distance: Math.round(dist), angle: Math.atan2(dy, dx) });
                            }
                        }
                    }
                    foodList.sort(function(a, b) { return a.distance - b.distance; });
                    result.foods = foodList.slice(0, 50);
                }
                
                if (window.slithers && window.slithers.length > 0) {
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && s.xx !== undefined && s !== mySnake && s.id !== mySnake.id) {
                            var head_dx = s.xx - sx, head_dy = s.yy - sy;
                            var head_dist = Math.sqrt(head_dx*head_dx + head_dy*head_dy);
                            var min_body_dist = head_dist, closest_x = s.xx, closest_y = s.yy;
                            
                            if (s.pts && s.pts.length > 0) {
                                for (var j = 0; j < s.pts.length; j++) {
                                    var pt = s.pts[j];
                                    if (pt && pt.xx !== undefined && pt.yy !== undefined) {
                                        var bdx = pt.xx - sx, bdy = pt.yy - sy;
                                        var bdist = Math.sqrt(bdx*bdx + bdy*bdy);
                                        if (bdist < min_body_dist) { min_body_dist = bdist; closest_x = pt.xx; closest_y = pt.yy; }
                                    }
                                }
                            }
                            
                            if (min_body_dist < 3000) {
                                var body_segments = [];
                                if (s.pts && s.pts.length > 0) {
                                    for (var k = 0; k < s.pts.length; k++) {
                                        var seg = s.pts[k];
                                        if (seg && seg.xx !== undefined && seg.yy !== undefined) body_segments.push({x: seg.xx, y: seg.yy});
                                    }
                                }
                                result.other_snakes.push({
                                    id: s.id, head_x: s.xx, head_y: s.yy, closest_x: closest_x, closest_y: closest_y,
                                    angle: s.ang || s.ehang || 0, speed: s.sp || 0, length: s.sct || 0,
                                    distance: Math.round(min_body_dist), head_distance: Math.round(head_dist),
                                    angle_to: Math.atan2(closest_y - sy, closest_x - sx), body_segments: body_segments
                                });
                            }
                        }
                    }
                    result.other_snakes.sort(function(a, b) { return a.distance - b.distance; });
                    result.other_snakes = result.other_snakes.slice(0, 15);
                }
                
                if (window.preys && window.preys.length > 0) {
                    for (var i = 0; i < window.preys.length; i++) {
                        var p = window.preys[i];
                        if (p && p.xx !== undefined) {
                            var dx = p.xx - sx, dy = p.yy - sy, dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < 2000) {
                                result.preys.push({ x: p.xx, y: p.yy, size: p.sz || 1, distance: Math.round(dist), angle: Math.atan2(dy, dx) });
                            }
                        }
                    }
                    result.preys.sort(function(a, b) { return a.distance - b.distance; });
                    result.preys = result.preys.slice(0, 20);
                }
                
                return result;
            """
            )
            return result
        except Exception as e:
            print(f"Error getting full state: {e}")
            return None

    def move_to_angle(self, angle_degrees):
        """Move cursor to a position at the specified angle from center."""
        self.previous_angle = self.current_angle
        self.current_angle = angle_degrees % 360
        angle_radians = math.radians(angle_degrees)

        offset_x = int(self.radius * math.cos(angle_radians))
        offset_y = int(-self.radius * math.sin(angle_radians))

        self.driver.execute_script(
            f"""
            if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                window.snake.ang = {angle_radians};
                window.snake.wang = {angle_radians};
                window.snake.eang = {angle_radians};
            }}
            window.xm = {offset_x}; window.ym = {offset_y};
            window.mouseX = window.innerWidth / 2 + {offset_x};
            window.mouseY = window.innerHeight / 2 + {offset_y};
            var event = new MouseEvent('mousemove', {{
                view: window, bubbles: true, cancelable: true,
                clientX: window.innerWidth / 2 + {offset_x}, clientY: window.innerHeight / 2 + {offset_y}
            }});
            document.dispatchEvent(event);
        """
        )

        self.capture_screenshot(action_type="move")

    def boost(self, enabled=True):
        """Toggle boost mode."""
        self.is_boosting = enabled
        angle_radians = math.radians(self.current_angle)
        offset_x = int(self.radius * math.cos(angle_radians))
        offset_y = int(-self.radius * math.sin(angle_radians))

        event_type = "mousedown" if enabled else "mouseup"
        self.driver.execute_script(
            f"""
            if (typeof window.snake !== 'undefined' && window.snake !== null) {{
                window.snake.ang = {angle_radians};
                window.snake.wang = {angle_radians};
                window.snake.eang = {angle_radians};
            }}
            window.xm = {offset_x}; window.ym = {offset_y};
            var canvas = document.querySelector('canvas');
            var event = new MouseEvent('{event_type}', {{ bubbles: true }});
            canvas.dispatchEvent(event);
        """
        )

        action_type = "boost_start" if enabled else "boost_end"
        self.capture_screenshot(action_type=action_type)

    def maintain_direction(self, duration=1.0, interval=0.05):
        """Maintain current direction for a specified duration."""
        steps = int(duration / interval)
        for _ in range(steps):
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
                window.xm = {offset_x}; window.ym = {offset_y};
            """
            )

            if self.record_video and self.frame_count % 2 == 0:
                self.capture_screenshot(action_type="hold")
            time.sleep(interval)

        self.capture_screenshot(action_type="hold")

    def capture_frame_only(self):
        """Capture a frame for video without logging."""
        if not self.record_video or not self.screenshot_folder:
            return

        try:
            screenshot_png = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            self.video_annotations.append(None)
            self.frame_count += 1
        except Exception as e:
            print(f"Error capturing frame: {e}")

    def is_game_over(self):
        """Check if the game over screen is displayed."""
        try:
            play_buttons = self.driver.find_elements(By.CLASS_NAME, "btnt")
            for button in play_buttons:
                if "Play" in button.text and button.is_displayed():
                    return True
            return False
        except Exception:
            return False
