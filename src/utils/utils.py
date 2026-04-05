import re

from src.config import ANSWER_TAG


def extract_answer(text: str) -> str:
    """Extract text from answer tags defined by ANSWER_TAG constant.

    Returns the content inside the tags, or empty string if not found.
    """
    match = re.search(rf'<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''


def get_completion_text(completion) -> str:
    """Extract text from a completion (handles both list and string formats)."""
    if isinstance(completion, list):
        return completion[0].get('content', '') if completion else ''
    return str(completion)


def find_starting_and_goal_positions(matrix):
    """Find start (*) and goal (#) positions in a maze matrix.

    Returns start_row, start_col, goal_row, goal_col.
    """
    start_row, start_col = None, None
    goal_row, goal_col = None, None
    for r, row in enumerate(matrix):
        for c, cell in enumerate(row):
            if cell == '*':
                start_row, start_col = r, c
            elif cell == '#':
                goal_row, goal_col = r, c
    return start_row, start_col, goal_row, goal_col