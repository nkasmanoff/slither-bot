# Tweet Thread: Training REINFORCE to Play Slither.io

🧵 1/18
I trained a REINFORCE agent to play Slither.io! 🐍

Instead of computer vision, I inject JavaScript to read the game's internal state and control the snake directly. The bot learns through trial and error using policy gradient RL.

Here's how it works 👇

---

🧵 2/18
The project has two approaches:

1️⃣ Rule-based: Hand-crafted logic (flee if enemy < 300 units, otherwise seek food)
2️⃣ REINFORCE: Neural network that learns from experience

Both use Selenium to control the browser and JavaScript injection to read game state.

---

🧵 3/18
Key insight: Slither.io stores everything in global JS variables!

-   `window.snake` = player's snake
-   `window.slithers` = all snakes (enemies)
-   `window.foods` = food pellets
-   `window.preys` = high-value food from dead snakes

We read these directly - no vision needed! 🎯

---

🧵 4/18
Critical improvement: Check the ENTIRE snake body for collisions, not just the head!

Collisions can happen with any body segment. We iterate through all `pts` (body points) to find the closest distance. Much safer than head-only detection.

---

🧵 5/18
Actions are executed by manipulating game variables:

```javascript
window.snake.ang = angle_radians;
window.xm = offset_x;
window.ym = offset_y;
document.dispatchEvent(mousemove_event);
```

8 discrete directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

---

🧵 6/18
State representation (11 dimensions):

-   Current angle & snake length
-   Nearest food distance & angle
-   Nearest prey distance & angle
-   Nearest enemy distance & angle
-   Counts of nearby foods/preys/enemies

All normalized to [-1, 1] for stable training.

---

🧵 7/18
What is REINFORCE? 🎓

A policy gradient algorithm that:

1. Plays an episode with current policy
2. Collects rewards
3. Computes returns (discounted future rewards)
4. Updates policy to increase probability of high-return actions

It directly optimizes the policy - no value function needed!

---

🧵 8/18
The policy network: Simple 3-layer MLP

Input: 11D state vector
Hidden: 128 → 128 (ReLU)
Output: 8 action probabilities (softmax)

Takes state → outputs probability distribution over actions → sample action → execute!

---

🧵 9/18
Reward design is crucial! 🎁

-   +10 per unit length increase (food collection)
-   -2.5 per step (survival penalty)
-   -50 for dying + 0.5 × final length

Balances immediate rewards (food) with long-term goals (survival & growth).

---

🧵 10/18
REINFORCE update step:

1. Compute discounted returns: R*t = r_t + γ·r*{t+1} + γ²·r\_{t+2} + ...
2. Normalize returns (subtract mean, divide by std) - reduces variance!
3. Policy loss: -log π(a|s) × R
4. Backprop & update

The math: ∇J(θ) = E[∇log π(a|s) × R]

---

🧵 11/18
Why normalization?

REINFORCE has HIGH variance. Baseline normalization (subtracting mean return) reduces variance without changing the expected gradient direction.

Makes training much more stable! 📈

---

🧵 12/18
Training process:

-   Online learning: Update after each episode
-   Each episode: Play until death or 1000 steps
-   Collect rewards & log-probabilities
-   Update policy using REINFORCE
-   Save best model when new max length achieved

---

🧵 13/18
What the agent learns:

Early: Avoid enemies (steer away from nearby snakes)
Mid: Seek food (navigate toward pellets)
Later: Balance exploration/exploitation (chase prey vs avoid danger)

Learning curve is noisy (high variance) but improves over time!

---

🧵 14/18
Comparison with rule-based policy:

Rule-based: Simple if-then logic (flee if enemy < 300 units)

RL agent potential:

-   Predict enemy movement patterns
-   Optimize paths to food while avoiding danger
-   Learn when to boost strategically

---

🧵 15/18
Challenges:

❌ High variance (single lucky/unlucky episode affects gradient)
✅ Mitigation: Baseline normalization + discount factor

❌ Sample inefficient (only updates after full episode)
💡 Future: PPO, Actor-Critic, experience replay

❌ Real-world noise (network latency, unpredictable players)

---

🧵 16/18
Key takeaways:

✅ State extraction > computer vision (work with internal game state!)
✅ Reward design shapes what agent learns
✅ REINFORCE is simple but effective (great starting point)
✅ Real-world RL is hard (variance, sample efficiency, noise)

---

🧵 17/18
Tech stack:

-   PyTorch (neural networks)
-   Gymnasium (environment interface)
-   Selenium (browser automation)
-   3-layer MLP (128 hidden units)
-   Adam optimizer (lr=0.001)
-   Discount factor γ=0.99

---

🧵 18/18
Full blog post with code examples, detailed explanations, and results: [link]

Code on GitHub: [link]

Train your own agent: `python slither_rl.py`

#ReinforcementLearning #MachineLearning #Python #PyTorch #REINFORCE #SlitherIO #RL #AI

---

## Alternative Shorter Version (12 tweets)

🧵 1/12
I trained REINFORCE to play Slither.io! 🐍

Instead of vision, I inject JavaScript to read game state directly. The bot learns through trial and error using policy gradient RL.

Thread on how it works 👇

---

🧵 2/12
Key insight: Slither.io stores state in global JS variables!

-   `window.snake` = player
-   `window.slithers` = enemies
-   `window.foods` = food pellets

We read these directly - no computer vision needed! 🎯

---

🧵 3/12
Critical: Check ENTIRE snake body for collisions, not just head!

We iterate through all body segments to find closest distance. Much safer collision avoidance.

---

🧵 4/12
State: 11D vector (angle, length, nearest food/prey/enemy distances & angles, counts)

Actions: 8 discrete directions (0°, 45°, 90°, ...)

Policy: 3-layer MLP (128 hidden units) → softmax over actions

---

🧵 5/12
REINFORCE = Policy gradient algorithm:

1. Play episode with current policy
2. Collect rewards
3. Compute returns (discounted future rewards)
4. Update policy: increase prob of high-return actions

Directly optimizes policy - no value function!

---

🧵 6/12
Reward design:

-   +10 per unit length increase
-   -2.5 per step
-   -50 for dying + 0.5 × final length

Balances immediate rewards (food) with long-term goals (survival).

---

🧵 7/12
REINFORCE update:

1. Compute discounted returns R*t = r_t + γ·r*{t+1} + ...
2. Normalize (subtract mean, divide by std) - reduces variance!
3. Loss: -log π(a|s) × R
4. Backprop & update

Math: ∇J(θ) = E[∇log π(a|s) × R]

---

🧵 8/12
Why normalization?

REINFORCE has HIGH variance. Baseline normalization reduces variance without changing expected gradient direction.

Makes training stable! 📈

---

🧵 9/12
Training: Online learning

-   Update after each episode
-   Play until death or 1000 steps
-   Collect rewards & log-probs
-   Update policy
-   Save best model on new max length

---

🧵 10/12
What agent learns:

Early: Avoid enemies
Mid: Seek food  
Later: Balance exploration/exploitation

Learning curve is noisy but improves over time!

---

🧵 11/12
Challenges:

❌ High variance → Baseline normalization
❌ Sample inefficient → Future: PPO, Actor-Critic
❌ Real-world noise → Network latency, unpredictable players

---

🧵 12/12
Takeaways:

✅ State extraction > vision
✅ Reward design matters
✅ REINFORCE simple but effective
✅ Real-world RL is hard

Full blog post: [link]
Code: [link]

#ReinforcementLearning #MachineLearning #Python #PyTorch #REINFORCE
