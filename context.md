This project integrates two main data sources: StatsBomb Events Data and Metrica Sports Tracking Data. Both datasets provide complementary information useful for AI models analyzing soccer matches.

---

1. StatsBomb Events Data

- Format: JSON (array of objects)

- Granularity: Event-based (specific match events with contextual details)

- Typical Event Example: "Starting XI", passes, shots, fouls, etc.

Main fields:

- id: Unique identifier of the event

- index: Sequential index of the event

- period: Match period (1 = first half, 2 = second half)

- timestamp, minute, second: Time indicators

- type: Event category (e.g., "Starting XI", "Pass", "Shot")

- possession: Possession sequence ID

- possession_team: Team in possession

- play_pattern: Context of play (e.g., "Regular Play", "From Corner")

- team: Team performing the event

- duration: Duration of the event (float in seconds)

- tactics: (only for Starting XI events) formation + list of players and their positions

Example – "Starting XI" event:

- Includes formation (e.g., 4-2-3-1)

- lineup: Array of players

  - Each player has:

    - player.id, player.name

    - position.id, position.name (e.g., Goalkeeper, Right Back)

    - jersey_number

This dataset is high-level, event-based and provides semantic information (who did what, when, and in which tactical context).

---

2. Metrica Sports Tracking Data

- Format: CSV

- Granularity: Frame-by-frame spatio-temporal tracking (player and ball positions)

Structure:

- Header row contains player identifiers (Player15, Player16, …, Player28) and columns for ball position

- Each record represents a single frame in the match with timestamp

Main fields:

- Period: Match half (1 = first half, 2 = second half)

- Frame: Frame number within the match video

- Time [s]: Timestamp in seconds (float, high precision)

- PlayerXX: X and Y coordinates of each player (two columns per player)

  - Example: Player15_x, Player15_y

- Ball: X and Y coordinates of the ball

Example row:

- Period = 1

- Frame = 1

- Time [s] = 0.04

- Player positions given as normalized coordinates (0–1 scale) across the pitch

- Ball position reported similarly

This dataset is low-level, continuous tracking data giving spatial and temporal movement of all players and the ball throughout the match.
