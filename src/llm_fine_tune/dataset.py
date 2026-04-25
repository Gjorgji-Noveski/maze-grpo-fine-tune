import json

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


def load_maze_dataset(path, tokenizer):
    """Load a prepared maze dataset and apply the tokenizer's chat template.

    Reads the JSON produced by prepare_dataset.py and adds a 'prompt' field
    to each entry. Returns the list of entries ready for training/eval.
    """
    with open(path) as f:
        entries = json.load(f)
    for entry in entries:
        entry['prompt'] = _apply_chat_template(tokenizer, entry['question'])
    return entries