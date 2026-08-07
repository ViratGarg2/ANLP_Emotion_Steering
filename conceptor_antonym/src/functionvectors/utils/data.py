import json
import pickle
import random


def load_input_output_pairs(path):
    """
    Loads input-output pairs from a JSON file located at the specified path.
    These pairs could represent any type of relational data, such as antonyms, country-capital, or uncapitalized-capitalized pairs.

    Args:
        path (str): The path to the JSON file containing the input-output pairs.

    Prints:
        The number of pairs loaded from the file.
    """
    with open(path, "r") as file:
        pairs = json.load(file)
        return pairs


def get_output(input_string, pairs):
    """
    Retrieves the corresponding output for a given input string from a list of input-output pairs.

    Args:
        input_string (str): The input string for which to find the corresponding output.
        pairs (list): The list of input-output pairs.

    Returns:
        str or None: The corresponding output string if found, otherwise None.
    """
    for pair in pairs:
        if pair["input"] == input_string:
            return pair["output"]
    return None


def get_input(output_string, pairs):
    """
    Retrieves the corresponding input for a given output string from a list of input-output pairs.

    Args:
        output_string (str): The output string for which to find the corresponding input.
        pairs (list): The list of input-output pairs.

    Returns:
        str or None: The corresponding input string if found, otherwise None.
    """
    for pair in pairs:
        if pair["output"] == output_string:
            return pair["input"]
    return None


def create_random_pairs_string(pairs, num_pairs):
    """
    Creates a string of randomly selected input-output pairs from a given list, with the last pair missing its output.

    Args:
        pairs (list): The list of input-output pairs to sample from.
        num_pairs (int): The number of pairs to include in the string.

    Returns:
        str: A string formatted with randomly selected pairs, where the last pair is missing its output.
    """
    sampled_pairs = random.sample(pairs, num_pairs)
    pairs_string = (
        ", ".join([f"{pair['input']}:{pair['output']}" for pair in sampled_pairs[:-1]])
        + f", {sampled_pairs[-1]['input']}:"
    )
    return pairs_string


def get_unique_random_inputs_formatted(pairs, n):
    """
    Returns a list of N unique input strings randomly sampled from the list of input-output pairs,
    with each input string formatted by adding a ':' at the end.

    Args:
        pairs (list): The list of input-output pairs to sample from.
        n (int): The number of unique input strings to return.

    Returns:
        list: A list of N unique input strings, each formatted with a ':' at the end.
    """
    unique_inputs = list(set(pair["input"] for pair in pairs))
    if len(unique_inputs) == 0:
        raise ValueError("No unique inputs available to sample from.")

    if len(unique_inputs) < n:
        sampled_inputs = random.choices(unique_inputs, k=n)
    else:
        sampled_inputs = random.sample(unique_inputs, n)

    formatted_inputs = [input_string + ":" for input_string in sampled_inputs]
    return formatted_inputs


def generate_experiment_data(
    model,
    num_experiments: int,
    steering_prompts_path: str,
    num_steering_prompts: int,
    num_steering_examples_per_prompt: int,
    num_input_prompts: int,
):
    """
    Generates data for multiple experiments, including prompts to steer and steering examples.

    Args:
        model: The model to use for generating steering examples.
        num_experiments (int): The number of experiments to generate data for.
        steering_prompts_path (str): The path to the JSON file containing the steering prompts.
        num_steering_prompts (int): The number of steering prompts to use for each experiment.
        num_steering_examples_per_prompt (int): The number of steering examples to generate for each steering prompt.
        num_input_prompts (int): The number of input prompts to test the model on.
    
    Returns:
        dict: A dictionary containing the data for each experiment, including prompts to steer and steering examples,
            with keys: "prompts_to_steer", "steering_examples", "correct_outputs_1st"
    """
    # Store separate prompts and steering examples for each experiment
    experiment_data = {}

    for exp in range(num_experiments):
        input_output_pairs = load_input_output_pairs(steering_prompts_path)
        steering_examples = []
        for _ in range(num_steering_prompts):
            steering_examples.append(
                create_random_pairs_string(
                    input_output_pairs, num_steering_examples_per_prompt
                )
            )
        prompts_to_steer = get_unique_random_inputs_formatted(
            input_output_pairs, num_input_prompts
        )

        correct_outputs_full = [
            get_output(prompt[:-1], input_output_pairs)
            for prompt in prompts_to_steer
        ]
        # NOTE: rename to correct_outputs_input_ids?
        correct_outputs_1st = [
            model.tokenizer.tokenize(out_i)[0]
            for out_i in correct_outputs_full
        ]

        experiment_data[exp] = {
            "prompts_to_steer": prompts_to_steer,
            "steering_examples": steering_examples,
            "correct_outputs_1st": correct_outputs_1st,
        }

    return experiment_data


def load_mean_activations(mean_activations_path: str, device: str = "cuda"):
    """
    Loads mean activations from a pickle file located at the specified path.

    Args:
        mean_activations_path (str): The path to the pickle file containing the mean activations.
        device (str): The device to move the tensors to (default: "cuda").

    Returns:
        dict: A dictionary containing the mean activations, with keys as layer indices and values as tensors.
    """
    with open(mean_activations_path, "rb") as file:
        mean_train_activations = pickle.load(file)

    mean_train_activations = {
        key: tensor.to(device) for key, tensor in mean_train_activations.items()
    }
    return mean_train_activations