import re
import random
from collections import Counter

from src.config import THINK_TAG, ANSWER_TAG
from src.utils.utils import extract_answer, get_completion_text, find_starting_and_goal_positions


DIRECTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}


def _simulate_directions(matrix, start, directions_text):
    """Simulate a sequence of directions on a maze grid.

    Returns (current_row, current_col, valid_steps, wall_hits).
    """
    current_row, current_col = start
    pred_dirs = directions_text.lower().strip().split()
    rows, cols = len(matrix), len(matrix[0])

    valid_steps = 0
    wall_hits = 0

    for direction in pred_dirs:
        if direction not in DIRECTIONS:
            wall_hits += 1
            continue

        dr, dc = DIRECTIONS[direction]
        new_row, new_col = current_row + dr, current_col + dc
        current_row, current_col = new_row, new_col

        in_bounds = 0 <= new_row < rows and 0 <= new_col < cols
        if in_bounds and matrix[new_row][new_col] != 'X':
            valid_steps += 1
        else:
            wall_hits += 1

    return current_row, current_col, valid_steps, wall_hits


def simulate_path(completions, metadata, **kwargs) -> list[float]:
    """Reward function using path simulation through the maze.

    Evaluates path mechanics:
    - valid steps taken
    - wall/boundary hits
    - reaching the goal
    - penalizes moves after reaching goal
    """
    step_reward = 0.5
    wall_penalty = -1.0
    reached_end_reward = 10.0
    overshoot_penalty = -2.0

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        text = extract_answer(get_completion_text(completion))
        start_row, start_col, goal_row, goal_col = find_starting_and_goal_positions(mat)

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        valid_steps = 0
        wall_hits = 0
        reached_end = False
        moves_after_reaching_end = 0
        rows, cols = len(mat), len(mat[0])

        for direction in pred_dirs:
            if direction not in DIRECTIONS:
                wall_hits += 1
                continue

            dr, dc = DIRECTIONS[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

            # Check validity
            in_bounds = 0 <= new_row < rows and 0 <= new_col < cols
            if in_bounds and mat[new_row][new_col] != 'X':
                valid_steps += 1
            else:
                wall_hits += 1

            # Check if at goal
            if current_row == goal_row and current_col == goal_col:
                reached_end = True
            elif reached_end:
                # Already reached goal but kept moving
                moves_after_reaching_end += 1

        score = (
            (valid_steps * step_reward) +
            (wall_hits * wall_penalty) +
            (reached_end_reward if reached_end else -3.0) +
            (moves_after_reaching_end * overshoot_penalty)
        )
        rewards.append(float(score))

    return rewards


def distance_reward(completions, metadata, **kwargs) -> list[float]:
    """Reward based on how much closer we got to the goal.

    Returns 0 if no final answer extracted (let format_reward handle that penalty).
    Otherwise, returns positive reward for getting closer, negative for moving away.
    """
    rewards = []
    for completion, meta in zip(completions, metadata):
        text = extract_answer(get_completion_text(completion))
        pred_dirs = text.lower().strip().split()

        # No answer extracted - return neutral, let format_reward penalize
        if not pred_dirs:
            rewards.append(0.0)
            continue

        mat = meta['matrix']
        start_row, start_col, goal_row, goal_col = find_starting_and_goal_positions(mat)

        # Calculate initial distance to goal (Manhattan)
        current_row, current_col = start_row, start_col
        initial_distance = abs(start_row - goal_row) + abs(start_col - goal_col)

        for direction in pred_dirs:
            if direction not in DIRECTIONS:
                continue
            dr, dc = DIRECTIONS[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
        # Distance improvement reward: positive if closer, negative if further
        distance_improvement = initial_distance - final_distance
        reward = 3.0 * (distance_improvement / initial_distance)

        rewards.append(float(reward))

    return rewards


def length_reward(completions, answer, **kwargs) -> list[float]:
    """Reward function that compares predicted length to ground truth length.

    - If lengths match: +1 reward
    - If lengths differ: 0 reward
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        text = extract_answer(get_completion_text(completion))

        pred_dirs = text.lower().strip().split()
        truth_dirs = ground_truth.lower().strip().split()

        if len(pred_dirs) == len(truth_dirs):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def no_answer_reward(completions, **kwargs) -> list[float]:
    """Penalize completions that have no extractable answer."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        answer = extract_answer(text)
        if len(answer.strip()) == 0:
            rewards.append(-5.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Reward proper tag structure in outputs.

    Checks for <think> and <answer> tags.
    """
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)

        count = 0.0
        if len(re.findall(rf"<{THINK_TAG}>", text, re.IGNORECASE)) == 1:
            count += 0.25
        if len(re.findall(rf"</{THINK_TAG}>", text, re.IGNORECASE)) == 1:
            count += 0.25
        if len(re.findall(rf"<{ANSWER_TAG}>", text, re.IGNORECASE)) == 1:
            count += 0.25
        if len(re.findall(rf"</{ANSWER_TAG}>", text, re.IGNORECASE)) == 1:
            count += 0.25
        rewards.append(count)

    return rewards


def got_to_end_reward(completions, metadata, **kwargs) -> list[float]:
    """Binary reward for reaching the goal."""
    reached_end_reward = 2.0

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        text = extract_answer(get_completion_text(completion))
        start_row, start_col, goal_row, goal_col = find_starting_and_goal_positions(mat)

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        for direction in pred_dirs:
            if direction not in DIRECTIONS:
                continue
            dr, dc = DIRECTIONS[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        # Check if at goal
        reached_end = current_row == goal_row and current_col == goal_col

        score = reached_end_reward if reached_end else 0.0
        rewards.append(float(score))

    return rewards


def binary_got_closer(completions, metadata, **kwargs) -> list[float]:
    """Binary reward: 0.5 if final position is closer to goal than start, else 0."""
    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        text = extract_answer(get_completion_text(completion))
        start_row, start_col, goal_row, goal_col = find_starting_and_goal_positions(mat)

        initial_dist = abs(start_row - goal_row) + abs(start_col - goal_col)

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        for direction in pred_dirs:
            if direction not in DIRECTIONS:
                continue
            dr, dc = DIRECTIONS[direction]
            new_row, new_col = current_row + dr, current_col + dc
            current_row, current_col = new_row, new_col

        final_dist = abs(current_row - goal_row) + abs(current_col - goal_col)
        score = 0.5 if final_dist < initial_dist else 0.0
        rewards.append(float(score))

    return rewards


def validity_reward(completions, **kwargs) -> list[float]:
    """Reward valid direction tokens in final answer.

    Returns 0 if no answer extracted, otherwise rewards based on
    ratio of valid directions (up, down, left, right) to total tokens.
    """
    valid_directions = {'up', 'down', 'left', 'right'}

    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        final_answer = extract_answer(text)
        tokens = final_answer.lower().strip().split()

        # No answer extracted - return neutral
        if not tokens:
            rewards.append(0.0)
            continue

        valid_count = sum(1 for t in tokens if t in valid_directions)
        validity_ratio = valid_count / len(tokens)

        # Reward high validity ratio (max 3)
        rewards.append(validity_ratio * 3.0)

    return rewards


def diversity_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward unique completions, penalize identical ones to prevent mode collapse.

    Uses random tiebreaking so identical completions get different rewards,
    creating gradient signal even when all outputs collapse to the same text.
    """
    texts = []
    for completion in completions:
        text = get_completion_text(completion)
        final = extract_answer(text)
        texts.append(final.lower().strip())

    counts = Counter(texts)
    n = len(texts)

    seen_counts = {}

    rewards = []
    for text in texts:
        frequency = counts[text] / n

        # Base penalty for duplicates
        base_reward = 1.0 - (2.0 * frequency)

        # Add decreasing reward for each duplicate instance
        # First occurrence gets base, subsequent ones get progressively worse
        if text not in seen_counts:
            seen_counts[text] = 0
        seen_counts[text] += 1

        # Tiebreaker: penalize later duplicates more heavily
        duplicate_penalty = -1.0 * (seen_counts[text] - 1) / n

        # Small random noise to break exact ties in gradient computation
        noise = random.uniform(-0.02, 0.02)

        rewards.append(base_reward + duplicate_penalty + noise)

    return rewards