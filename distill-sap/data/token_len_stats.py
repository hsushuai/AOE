import json
import os
import traceback
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_token_stats(jsonl_path: str, tokenizer_name: str, model_length: int=None) -> dict:
    """
    Compute the total token count for each sample in the given jsonl dataset and return statistics.

    Args:
        jsonl_path (str): The path to the input train.jsonl file.
        tokenizer_name (str): The name of the tokenizer to use for tokenization.
        model_length (int, optional): The maximum length of the model. Defaults to the tokenizer's max length.

    Returns:
        dict: A dictionary containing the mean, std, min, max, and size statistics.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_length = model_length or tokenizer.model_max_length
    token_lengths = []
    num_exceed = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(full_text, add_special_tokens=False)
            token_lengths.append(len(tokens["input_ids"]))
            if token_lengths[-1] > model_length:
                num_exceed += 1

    lengths = np.array(token_lengths)

    stats = {
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "size": int(len(lengths)),
        "num_exceed": num_exceed
    }

    return stats


def main(data_dir: str, tokenizer_name: str, model_length: int=None):
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {}
        for filename in filenames:
            jsonl_path = os.path.join(data_dir, filename)
            futures[executor.submit(compute_token_stats, jsonl_path, tokenizer_name, model_length)] = filename
        for future in as_completed(futures):
            filename = futures[future]
            try:
                stats = future.result()
                tqdm.write("=" * 40)
                tqdm.write(f"Statistics for {filename}:")
                tqdm.write(f"Mean:   {stats['mean']}")
                tqdm.write(f"Std:    {stats['std']}")
                tqdm.write(f"Min:    {stats['min']}")
                tqdm.write(f"Max:    {stats['max']}")
                tqdm.write(f"Size:   {stats['size']}")
                tqdm.write(f"Exceed: {stats['num_exceed']}")
            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")
                tqdm.write(traceback.format_exc())
            tqdm.write("=" * 40)


if __name__ == "__main__":
    main("distill-sap/data/data-1", "/mnt/opsstorage/models/Qwen3-4B", model_length=4000)