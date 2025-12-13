import json
import math
import os
import re
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By


class SlitherController:
    def __init__(self, driver, save_screenshots=False):
        self.driver = driver
        self.radius = 50  # pixels from center
        self.current_angle = 0
        self.previous_angle = 0
        self.is_boosting = False
        self.save_screenshots = save_screenshots
        self.frame_count = 0
        self.screenshot_folder = None
        self.images_folder = None
        self.game_start_time = datetime.now()

        # Game state log for JSON export
        self.game_log = {"game_start": self.game_start_time.isoformat(), "frames": []}

        if self.save_screenshots:
            self._setup_screenshot_folder()

    def _setup_screenshot_folder(self):
        """Create the games folder and timestamped subfolder for screenshots."""
        games_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
        os.makedirs(games_folder, exist_ok=True)

        # Create subfolder with game start timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.screenshot_folder = os.path.join(games_folder, timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)

        # Create images subfolder for screenshots
        self.images_folder = os.path.join(self.screenshot_folder, "images")
        os.makedirs(self.images_folder, exist_ok=True)
        print(f"Screenshots will be saved to: {self.images_folder}")

    def capture_screenshot(self, action_type="move"):
        """Capture a screenshot with metadata and log game state."""
        if not self.save_screenshots or not self.screenshot_folder:
            return

        boost_str = "boost" if self.is_boosting else "normal"
        filename = f"frame_{self.frame_count:06d}_angle_{self.current_angle:.1f}_speed_{boost_str}.png"
        filepath = os.path.join(self.images_folder, filename)

        # IMPORTANT: Capture game state FIRST, then screenshot immediately after
        # This minimizes timing mismatch between state and image
        detailed_state = self.get_detailed_state()
        snake_length = self.get_snake_length()
        snake_rank = self.get_snake_rank()

        # Take screenshot IMMEDIATELY after getting state
        self.driver.save_screenshot(filepath)

        # Now record timestamp (after capture)
        current_time = datetime.now()
        elapsed_time = (current_time - self.game_start_time).total_seconds()

        # Create frame state entry
        frame_state = {
            "frame_id": self.frame_count,
            "timestamp": current_time.isoformat(),
            "elapsed_seconds": round(elapsed_time, 3),
            "image_file": f"images/{filename}",
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

        # Add previous frames context (last 5 actions)
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

        # Auto-save JSON every 10 frames
        if self.frame_count % 10 == 0:
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

        json_path = os.path.join(self.screenshot_folder, "game_log.json")
        with open(json_path, "w") as f:
            json.dump(self.game_log, f, indent=2)

        return json_path

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
                if (window.slithers && window.slithers.length > 0) {
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s && s.xx !== undefined && s !== mySnake && s.id !== mySnake.id) {
                            var dx = s.xx - sx;
                            var dy = s.yy - sy;
                            var dist = Math.sqrt(dx*dx + dy*dy);
                            if (dist < 3000) {  // Only snakes within range
                                result.other_snakes.push({
                                    id: s.id,
                                    x: s.xx,
                                    y: s.yy,
                                    angle: s.ang || s.ehang || 0,
                                    speed: s.sp || 0,
                                    length: s.sct || 0,
                                    distance: Math.round(dist),
                                    angle_to: Math.atan2(dy, dx)
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
            time.sleep(interval)

        # Take a screenshot at the end
        self.capture_screenshot(action_type="hold")

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
    # Setup driver (assuming you already have this)
    driver = webdriver.Chrome()
    driver.get("http://slither.io")

    # Wait for game to load and start
    time.sleep(5)  # Adjust as needed, or use explicit waits
    # start the game
    play_button = driver.find_elements(By.CLASS_NAME, "btnt")
    for button in play_button:
        if "Play" in button.text:
            button.click()
            break
    print("Game started")
    time.sleep(1)

    # Create controller with screenshot saving enabled
    controller = SlitherController(driver, save_screenshots=True)

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
    print(f"Game session complete. {controller.frame_count} screenshots saved.")
    print(f"Game log saved to: {log_path}")
    driver.quit()
