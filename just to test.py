import ast
import re
def extract_final_answer(text: str) -> str:
    """Extract text from <final_answer></final_answer> tags.

    Returns the content inside the tags, or empty string if not found.
    """
    match = re.search(r'<final_answer>(.*?)</final_answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''

def simulate_path(prompts, completions, metadata, **kwargs) -> list[float]:
    """Reward function using path simulation through the maze.

    Key insight: We add distance-based shaping so the model gets gradient signal
    even when it doesn't reach the goal. This prevents mode collapse by rewarding
    outputs that get CLOSER to the goal, not just those that reach it.
    # Currently this does 4 things:
    - simulates path
    - calculates reward for how much we've come closer to the goal or if we ran away from it
    - reward for reaching end
    - penalizes moves after reaching goal
    """
    step_reward = 0.5
    wall_penalty = -1.0
    reached_end_reward = 10.0
    overshoot_penalty = -2.0  # Penalty per move after reaching goal

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    rewards = []
    for completion, meta in zip(completions, metadata):
        mat = meta['matrix']
        if isinstance(completion, list):
            text = completion[0].get('content', '') if completion else ''
        else:
            text = str(completion)

        # Extract only the final answer for evaluation
        text = extract_final_answer(text)

        # Find start and goal positions
        start_row, start_col = None, None
        goal_row, goal_col = None, None
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell == '*':
                    start_row, start_col = r, c
                elif cell == '#':
                    goal_row, goal_col = r, c

        current_row, current_col = start_row, start_col
        pred_dirs = text.lower().strip().split()

        valid_steps = 0
        wall_hits = 0
        reached_end = False
        moves_after_reaching_end = 0
        rows, cols = len(mat), len(mat[0])

        # Calculate initial distance to goal (Manhattan)
        initial_distance = abs(start_row - goal_row) + abs(start_col - goal_col)

        for direction in pred_dirs:
            if direction not in directions:
                wall_hits += 1
                continue

            dr, dc = directions[direction]
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

        # Calculate final distance to goal
        final_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

        # Distance improvement reward: positive if we got closer, negative if further
        # Scale by initial distance to normalize across different maze sizes
        distance_improvement = initial_distance - final_distance
        distance_reward = 3.0 * (distance_improvement / initial_distance)

        score = (
            (valid_steps * step_reward) + # 0
            (wall_hits * wall_penalty) + # 0
            distance_reward + # 0
            (reached_end_reward if reached_end else -3.0) + # HERE
            (moves_after_reaching_end * overshoot_penalty)
        )
        rewards.append(float(score))

    return rewards


test_query = """You are an expert maze solver. Your task is to find the shortest path from the start to the destination point in a grid.

First, think through your solution step by step inside <scratchpad></scratchpad> tags. You can use this area to analyze the maze, plan your path, and work through the problem.

Then, output only the sequence of directions (up, down, left, right) inside <final_answer></final_answer> tags.

Example format:
<scratchpad>
[Your reasoning here]
</scratchpad>
<final_answer>
up right down left
</final_answer>

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.

You may only move in four directions: up, down, left, and right.

Your output should be a sequence of directions that leads from * to #, e.g. right right down down up left

Now, find the shortest path from * to # in the following grid:
O # O
O * O
O X O
"""

completion = """<scratchpad>
The grid is:
O # O
O * O
O X O

The start is at * and the destination is at #.
We need to find the shortest path.
We can start by exploring the grid.
From the starting point (*), we can move right. The cell to the right is blocked (X).
So we must move down.
From the cell below the starting point, we can move right. The cell to the right is blocked (X).
So we must move left.
From the cell to the left, we can move down. The cell below is blocked (X).
So we must move right.
From the cell to the right, we can move down. The cell below is the destination (#).

So the path is down, right, right, down.
Let's verify this path:
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Right: (1, 2) -> (1, 3) - blocked
So, let's try another path.
1. Down: (0, 1) -> (1, 1)
2. Left: (1, 1) -> (1, 0)
3. Down: (1, 0) -> (2, 0)
4. Right: (2, 0) -> (2, 1)
5. Right: (2, 1) -> (2, 2) - blocked

Let's consider another approach:
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Down: (1, 2) -> (2, 2) - blocked
So we have to go back to (1, 1).

Let's consider the path: down, right, down, right.
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Down: (1, 2) -> (2, 2) - X
We have to go back to (1, 1). So let's try another path.

Let's try down, right, left, down.
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Left: (1, 2) -> (1, 1)
4. Down: (1, 1) -> (2, 1) - X
Doesn't work.

Let's try down, left, down, right.
1. Down: (0, 1) -> (1, 1)
2. Left: (1, 1) -> (1, 0)
3. Down: (1, 0) -> (2, 0) - X
Doesn't work.

Let's try down, right, down, left.
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Down: (1, 2) -> (2, 2) - X
Doesn't work.

Let's try down, down, right, right.
1. Down: (0, 1) -> (1, 1)
2. Down: (1, 1) -> (2, 1) - X
Doesn't work.

Let's try down, right, up, right. Doesn't work since there's no up.

Let's try down, right, down, right.
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Down: (1, 2) -> (2, 2) - X
Let's try: down, left, down, right.

Let's try down, right, down, right.
1. Down: (0, 1) -> (1, 1)
2. Right: (1, 1) -> (1, 2)
3. Down: (1, 2) -> (2, 2) - X. This is wrong.

The shortest path is down, right"""
metu = ast.literal_eval("{'source_dataset': 'shortest_path','source_index': 0,'matrix': [['O', '#', 'O'], ['O', '*', 'O'], ['O', 'X', 'O']],'solution': ['up'],'difficulty': {'rows': (3, 3), 'cols': (3, 3)}}")
simulate_path(prompts=[test_query], completions=[completion], metadata=[metu])
