# ⬡ Catan vs AI — Interactive Web Interface

A browser-based Catan game where you play 1-on-1 against a trained Reinforcement Learning bot. The board is rendered in real time as an SVG, your moves are sent to a Flask backend, and the bot responds using a **MaskablePPO** neural network trained with [catanatron-syslab](https://github.com/Zawiop/catanatron-syslab).

---

## What it looks like

- Full Catan board rendered as SVG — 19 terrain tiles, 54 intersections, 72 edges, and 9 ports with emoji trade-rate badges
- Left panel: your resources, victory points, dev card breakdown, longest road / largest army badges, and the game log
- Right panel: the bot's stats and your action buttons for the current turn
- Header: turn indicator, live dice roll display (⚄⚂ = 7), and New Game button

---

## How it works

The project has two main files — a Python backend and a single-page HTML frontend — that talk to each other through three JSON endpoints.

### `app.py` — Flask backend

Runs on your machine and handles everything game-related:

| Section | What it does |
|---|---|
| **Model loading** | Loads `FFF38MR2.zip` (the PPO model) and `FFF38MR2.pkl` (the VecNormalize stats) once at startup |
| **Geometry** | Converts cube coordinates `(q, r, s)` → pixel `(x, y)` for all 54 nodes and 19 tiles |
| **`init_game()`** | Creates a fresh 2-player Catan game (you as BLUE, bot as RED) |
| **`bot_decide()`** | Builds the observation, applies the adapter, normalises it, masks illegal moves, and asks the model for its action |
| **`serialize_state()`** | Converts the full game state into a JSON dict the browser can understand |
| **Routes** | `POST /api/new_game` · `GET /api/state` · `POST /api/action` |

After every human action, `_run_bot_turns()` loops until it's your turn again, so the bot plays out all of its moves (roll, build, end turn, etc.) in a single response.

### `adapter.py` — 4-player → 2-player bridge

The bot was trained on 4-player games, which produce a **1002-feature** observation vector. A 2-player game only generates **614 features** — the other 388 are slots for players P2 and P3.

`adapter.py` solves this by zero-filling the missing slots: for every feature name the 4-player model expects, it takes the value from the 2-player game if it exists, and uses `0.0` otherwise. This is correct because the model learned that all-zero P2/P3 features simply means those players have nothing built — which is always true in a 2-player game.

```
614 features from 2-player game
+
388 zeros for ghost players P2 and P3
= 1002-feature vector the model expects
```

### `templates/index.html` — Frontend

A single HTML file with no external JavaScript frameworks:

| Part | What it does |
|---|---|
| **CSS Grid** | 3-column layout: left panel (240px) · SVG board (fills remaining space) · right panel (240px) |
| **SVG layers** | 10 named `<g>` groups drawn back-to-front: ocean → sand → tiles → ports → roads → click targets → nodes → buildings → tokens → robber |
| **`drawBoard(state)`** | Redraws the entire SVG every turn from the JSON state |
| **`ui(state)`** | Updates resource counts, VP numbers, badges, log, and action buttons |
| **`doActions(state)`** | Builds the right-panel buttons; simple actions get direct-click buttons, board-placement actions (settlement, road, city, robber) toggle a `MODE` and highlight valid spots |
| **`submit(idx)`** | POSTs your chosen `action_idx` to `/api/action` and calls `ui()` on the response |

The key constant `TC = 72` in the JavaScript must always match `HEX_SIZE = 72` in `app.py` — both define the pixel scale of the hex grid.

---

## File structure

```
CatanWebsite/
├── app.py              # Flask backend — game engine, bot inference, API routes
├── adapter.py          # Pads 2-player obs to 1002 features for the 4-player model
├── FFF38MR2.zip        # Trained MaskablePPO model weights
├── FFF38MR2.pkl        # VecNormalize statistics (mean/variance for obs scaling)
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # Complete frontend — CSS, SVG board, JavaScript
```

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/professorcool09/CatanWebsite.git
cd CatanWebsite
```

**2. Install dependencies**
```bash
pip install flask catanatron catanatron_gym sb3-contrib numpy
```

Or if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

**3. Run the server**
```bash
python app.py
```

**4. Open your browser**
```
http://localhost:5000
```

Click **NEW GAME** and start playing.

---

## How to play

1. **Setup phase** — You and the bot take turns placing 2 settlements and 2 roads. The bot plays its setup turns automatically before handing control back to you.
2. **Your turn** — Click **Roll Dice** in the action panel. Resources are distributed, the dice roll is shown in the header.
3. **Build or trade** — Click action buttons on the right. For settlements, roads, cities, and robber moves, click the button to activate the mode, then click the highlighted spot directly on the board.
4. **End turn** — Click **End Turn**. The bot plays all of its moves and the board updates.
5. **Win condition** — First to 10 victory points wins.

### Victory points
| Thing | Points |
|---|---|
| Settlement | 1 VP |
| City | 2 VP |
| Longest Road (≥5 segments) | 2 VP |
| Largest Army (≥3 knights played) | 2 VP |
| Victory Point dev card | 1 VP each |

### Maritime trading
Trade ratios shown on the right panel. Port badges on the board show the resource and rate (2:1 for specific resources, ⚓ 3:1 for generic ports). Without a port, any 4 identical resources trade for 1 of any type.

---

## The bot

The RL bot was trained using **MaskablePPO** from `sb3-contrib` on the [Catanatron](https://github.com/Zawiop/catanatron-syslab) Gymnasium environment. For full details on training, reward functions, and evaluation results, see the training repository:

**[github.com/Zawiop/catanatron-syslab](https://github.com/Zawiop/catanatron-syslab)**

Key facts about the model used here (`FFF38MR2`):

- Trained on **4-player** games for approximately 20 million timesteps
- Observation space: **1002 features** (board layout, resources, buildings, roads, ports, robber position, VP counts for all 4 players)
- Action space: **290 discrete actions** (settlements, cities, roads, dev cards, trading, robber, roll, end turn)
- Action masking ensures the bot never attempts an illegal move
- Adapted to run in a 2-player game via `adapter.py` — the 388 unused P2/P3 feature slots are zero-filled

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| Game engine | Catanatron |
| RL model | MaskablePPO (sb3-contrib) · Stable Baselines 3 |
| Observation processing | NumPy · catanatron_gym |
| Frontend | Vanilla HTML · CSS Grid · SVG · Vanilla JavaScript |
| Fonts | Google Fonts (Bebas Neue · DM Sans) |

---

## Authors

Arrush Shah · Suraj Kidiyoor — Syslab P3, Dr. Yilmaz
