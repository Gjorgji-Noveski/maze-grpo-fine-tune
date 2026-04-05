import reasoning_gym
from llm_fine_tune.utils.config import SYSTEM_PROMPT


def _clean_prompt(question):
    """Remove infeasibility text and 'the length of' from maze prompts."""
    question = question.replace("the length of ", "")
    question = question.replace(
        "Your task is to find the shortest path from the start to the destination point in a grid.\n", "")
    start_idx = question.find('If there is no path')
    end_idx = question.find('Your output should be')
    return question[:start_idx] + question[end_idx:]


def _apply_chat_template(tokenizer, content):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def create_maze_dataset(tokenizer, size, min_rows, max_rows, min_cols, max_cols, p_blocked, seed):
    """Create a maze dataset, filtering infeasible entries and formatting prompts.

    Returns:
        tuple: (filtered_dataset, original_size, filtered_size)
    """
    dataset = reasoning_gym.create_dataset(
        "shortest_path",
        min_rows=min_rows,
        max_rows=max_rows,
        min_cols=min_cols,
        max_cols=max_cols,
        p_blocked=p_blocked,
        size=size,
        seed=seed
    )

    original_size = len(dataset)

    filtered = []
    for entry in dataset:
        if entry['answer'].lower() != 'infeasible':
            clean_prompt = _clean_prompt(entry['question'])
            entry['prompt'] = _apply_chat_template(tokenizer, clean_prompt)
            filtered.append(entry)

    return filtered, original_size, len(filtered)