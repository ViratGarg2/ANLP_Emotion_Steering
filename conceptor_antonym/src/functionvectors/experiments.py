import math
import pandas as pd
import os
from tqdm import tqdm
from utils.steering import (
    top_1_first_tokens,
    generate_conceptor_hook,
    generate_conceptor_hook_paper,
    generate_conceptor_hook_mean_centered,
    generate_ave_hook_addition,
    generate_ave_hook_addition_mean_centered,
)


def _get_model_id(model):
    """Helper function to get the model ID from a HookedTransformer model."""
    try:
        model_id = f"{model.name_or_path}_" if hasattr(model, "name_or_path") else ""
        model_id = f"{model.cfg.model_name}_"
    except:
        model_id = ""
    return model_id


def _get_filename(results_path: str, model_id: str, task: str, exp_type: str, fname_suffix: str = ""):
    """Helper function to get the filename for the results CSV file."""
    fname_suffix = f"_{fname_suffix}" if fname_suffix else ""
    fname_base = f"{results_path}/{model_id}{task}_{exp_type}{fname_suffix}"
    fname = f"{fname_base}.csv"
    fname_idx = 0
    while os.path.exists(fname):
        fname_idx += 1
        fname = f"{fname_base}_{fname_idx}.csv"
    return fname


def run_baseline_experiments(
    model,
    experiment_data: dict,
    num_experiments: int,
    task: str,
    results_path: str,
    save_results: bool = True,
):
    print("Calculating baseline accuracy...")
    baseline_success_count = 0
    total_prompts_for_baseline = 0

    # get the file name for the results CSV file
    model_id = _get_model_id(model)
    fname = _get_filename(results_path, model_id, task, "baseline")
    print(f"Storing results to `{fname}`")

    batch_size = 50
    baseline_data = []

    for exp in range(num_experiments):
        success_count = 0
        total_prompts = 0

        # Retrieve experiment-specific prompts and correct outputs
        prompts_to_steer = experiment_data[exp]["prompts_to_steer"]
        correct_outputs_1st = experiment_data[exp]["correct_outputs_1st"]

        num_batches = math.ceil(len(prompts_to_steer) / batch_size)

        for batch_idx in range(num_batches):
            batch_prompts = prompts_to_steer[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_correct_outputs_1st = correct_outputs_1st[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]

            # Generate baseline outputs from input prompts
            top_1_tokens = top_1_first_tokens(model, batch_prompts, fwd_hooks=[])

            # Increment success count if top-1 output matches the correct output
            for i, top_1_token in enumerate(top_1_tokens):
                if top_1_token == batch_correct_outputs_1st[i]:
                    success_count += 1
                total_prompts += 1

        baseline_accuracy = (success_count / total_prompts) * 100
        print(
            f"Experiment {exp + 1} Baseline Accuracy: {baseline_accuracy:.2f}% ({success_count}/{total_prompts} samples)"
        )

        # Store results for this experiment
        baseline_data.append(
            {
                "experiment": exp + 1,
                "success_count": success_count,
                "total_prompts": total_prompts,
                "baseline_accuracy": baseline_accuracy,
            }
        )

        # Update overall baseline success count and total prompts
        baseline_success_count += success_count
        total_prompts_for_baseline += total_prompts

    # Calculate and print the overall baseline accuracy
    overall_baseline_accuracy = (baseline_success_count / total_prompts_for_baseline) * 100
    print(
        f"Overall Baseline Unsteered Accuracy: {overall_baseline_accuracy:.2f}% ({baseline_success_count}/{total_prompts_for_baseline} samples)"
    )

    # Store the overall results
    baseline_data.append(
        {
            "experiment": "Overall",
            "success_count": baseline_success_count,
            "total_prompts": total_prompts_for_baseline,
            "baseline_accuracy": overall_baseline_accuracy,
        }
    )

    if save_results:
        baseline_df = pd.DataFrame(baseline_data)
        baseline_df.to_csv(fname, index=False)


def run_all_conceptor_experiments(
    model,
    experiment_data,
    conceptors_cache,
    configs_conceptoring,
    save_results,
    num_experiments: int,
    results_path: str,
    task: str,
    batch_size: int,
    mean_train_activations=None,
    verbose: bool = False,
    fname_suffix: str = "",
    pure: bool = False,
):
    results_conceptoring = []
    # get the file name for the results CSV file
    model_id = _get_model_id(model)
    # "_paper" -> h' = beta*C*h (arXiv:2410.16314 eq. 4)
    # "_mc"    -> bias-recovery form with the training mean
    mc = "_paper" if pure else ("_mc" if mean_train_activations is not None else "")
    fname = _get_filename(results_path, model_id, task, f"conceptor{mc}", fname_suffix)
    print(f"Storing results to `{fname}`")

    pbar = tqdm(enumerate(configs_conceptoring), total=len(configs_conceptoring), desc="Conceptor Steering")
    for config_idx, config in pbar:
        # Extract current experimental configurations
        layer = config.extraction_layer
        beta = config.beta
        aperture = config.aperture
        lambda_ = config.lambda_
        config_key = f"Layer {layer}, Beta {beta}, Aperture {aperture}, Lambda {lambda_}"
        if verbose:
            print(f"-----------{config_key}-----------")

        # Run the conceptor steering experiment
        results_conceptoring += run_conceptor_experiment(
            model=model,
            experiment_data=experiment_data,
            conceptors_cache=conceptors_cache,
            layer=layer,
            beta=beta,
            aperture=aperture,
            lambda_=lambda_,
            num_experiments=num_experiments,
            batch_size=batch_size,
            mean_train_activations=mean_train_activations,
            verbose=verbose,
            pure=pure,
        )
        final_acc = results_conceptoring[-1]['final_accuracy']
        pbar.set_postfix_str(f"{config_key} | {final_acc:.1%}")

        if save_results and (config_idx % 5 == 0 or config_idx == (len(configs_conceptoring) - 1)):
            df = pd.DataFrame(results_conceptoring)
            df.to_csv(fname, index=False)
    
    return results_conceptoring


def run_conceptor_experiment(
    model,
    experiment_data: dict,
    conceptors_cache: dict,
    layer: int,
    beta: float,
    aperture: float,
    num_experiments: int,
    batch_size: int,
    lambda_: float = 0.5,
    mean_train_activations = None,
    verbose: bool = False,
    pure: bool = False,
):
    return_data = []

    for exp in range(num_experiments):
        # Retrieve precomputed conceptor from the cache
        conceptor = conceptors_cache[exp][(layer, aperture)].cuda()

        success_count = 0
        total_prompts = 0

        # Retrieve experiment-specific prompts and correct outputs
        prompts_to_steer = experiment_data[exp]["prompts_to_steer"]
        correct_outputs_1st = experiment_data[exp]["correct_outputs_1st"]

        num_batches = math.ceil(len(prompts_to_steer) / batch_size)

        for batch_idx in range(num_batches):
            batch_prompts = prompts_to_steer[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_correct_outputs_1st = correct_outputs_1st[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]

            # Initialize hooks that will allow for the conceptor to be applied
            if pure:
                # h' = beta*C*h, the published form (arXiv:2410.16314 eq. 4).
                # lambda plays no part here.
                conceptor_hook = generate_conceptor_hook_paper(conceptor, beta)
            elif mean_train_activations is not None:
                conceptor_hook = generate_conceptor_hook_mean_centered(
                    conceptor=conceptor,
                    mean_train=mean_train_activations[layer],
                    beta=beta,
                    lambda_=lambda_,
                )
            else:
                conceptor_hook = generate_conceptor_hook(conceptor, beta, lambda_=lambda_)

            activation_modification = (f"blocks.{layer}.hook_resid_pre", conceptor_hook)
            editing_hooks = [activation_modification]

            # Generate steered outputs from input prompts using the conceptor hooks
            top_1_tokens = top_1_first_tokens(
                model, batch_prompts, fwd_hooks=editing_hooks
            )

            # Increment success count if top-1 output matches the correct output
            for i, top_1_token in enumerate(top_1_tokens):
                if top_1_token == batch_correct_outputs_1st[i]:
                    success_count += 1
                total_prompts += 1

        final_accuracy = success_count / total_prompts
        return_data.append({
            "layer": layer,
            "beta": beta,
            "aperture": aperture,
            "lambda_": lambda_,
            "experiment": exp + 1,
            "success_count": success_count,
            "total_prompts": total_prompts,
            "final_accuracy": final_accuracy,
        })
        model.reset_hooks()

        if verbose:
            print(f"Experiment {exp+1}: Final Accuracy: {final_accuracy:.2%} ({success_count}/{total_prompts} samples)")

    # Calculate average final accuracy across all experiments
    return_data.append({
        "layer": layer,
        "beta": beta,
        "aperture": aperture,
        "lambda_": lambda_,
        "experiment": "Average",
        "final_accuracy": sum([e["final_accuracy"] for e in return_data]) / num_experiments,
    })
    if verbose:
        print(f"Avg Final Accuracy: {return_data[-1]['average_final_accuracy']:.2%} over {num_experiments} experiments")

    return return_data


def run_all_actadd_experiments(
    model,
    experiment_data,
    averaged_activations_cache,
    configs_averaging,
    save_results,
    num_experiments: int,
    results_path: str,
    task: str,
    batch_size: int,
    mean_train_activations=None,
    verbose: bool = False,
):
    """
    Run all experiments for the addition steering method.

    Args:
        model (torch.nn.Module): The model to be used for the experiments.
        experiment_data (dict): The data for each experiment.
        averaged_activations_cache (dict): The precomputed averaged activations for each experiment.
        configs_averaging (List[ExperimentConfigAveraging]): The configurations for the experiments.
        save_results (bool): Whether to save the results to a CSV file.
        num_experiments (int): The number of experiments.
        results_path (str): The path to save the results.
        task (str): The task being performed.
        batch_size (int): The batch size for the experiments.
        mean_train_activations (Optional[dict]): The mean activations for each layer.
        verbose (bool): Whether to print verbose output.
    """
    results = []

    # get the file name for the results CSV file
    model_id = _get_model_id(model)
    mc = "_mc" if mean_train_activations is not None else ""
    fname = _get_filename(results_path, model_id, task, f"addition{mc}")
    print(f"Storing results to `{fname}`")

    pbar = tqdm(enumerate(configs_averaging), total=len(configs_averaging), desc="ActAdd Steering")
    for config_idx, config in pbar:
        # Extract current experimental configurations
        layer = config.extraction_layer
        beta = config.beta
        config_key = f"Layer {layer}, Beta {beta}"
        if verbose:
            print(f"-------------------------{config_key}-------------------------")

        for exp in range(num_experiments):
            # Retrieve precomputed averaged activations from the cache
            avg_activations = averaged_activations_cache[exp][layer]
            if mean_train_activations is not None:
                mean_train = mean_train_activations[layer]

            success_count = 0
            total_prompts = 0

            # Retrieve experiment-specific prompts and correct outputs
            prompts_to_steer = experiment_data[exp]["prompts_to_steer"]
            correct_outputs_1st = experiment_data[exp]["correct_outputs_1st"]

            num_batches = math.ceil(len(prompts_to_steer) / batch_size)

            for batch_idx in range(num_batches):
                batch_prompts = prompts_to_steer[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                batch_correct_outputs_1st = correct_outputs_1st[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                # Initialize hooks that will allow for the average vector to be added
                if mean_train_activations is None:
                    ave_hook = generate_ave_hook_addition(
                        steering_vector=avg_activations, beta=beta
                    )
                else:
                    ave_hook = generate_ave_hook_addition_mean_centered(
                        steering_vector=avg_activations, mean_train=mean_train, beta=beta
                    )
                activation_modification = (f"blocks.{layer}.hook_resid_pre", ave_hook)
                editing_hooks = [activation_modification]

                # Generate steered outputs from input prompts using the average steering hooks
                top_1_tokens = top_1_first_tokens(
                    model, batch_prompts, fwd_hooks=editing_hooks
                )

                # Increment success count if top-1 output matches the correct output
                for i, top_1_token in enumerate(top_1_tokens):
                    if top_1_token == batch_correct_outputs_1st[i]:
                        success_count += 1
                    total_prompts += 1

            final_accuracy = success_count / total_prompts
            results.append({
                "layer": layer,
                "beta": beta,
                "experiment": exp + 1,
                "success_count": success_count,
                "total_prompts": total_prompts,
                "final_accuracy": final_accuracy,
            })
            model.reset_hooks()

            if verbose:
                print(
                    f"Experiment {exp+1}: Final Accuracy: {final_accuracy:.2%}",
                    f"({success_count}/{total_prompts} samples)"
                )

        # Calculate average final accuracy across all experiments
        avg_final_acc = sum([r["final_accuracy"] for r in results[-num_experiments:]]) / num_experiments
        results.append({
            "layer": layer,
            "beta": beta,
            "experiment": "Average",
            "final_accuracy": avg_final_acc,
        })
        if verbose:
            print(
                f"Average Final Accuracy: {avg_final_acc:.2%} over {num_experiments} experiments"
            )

        pbarstr = f"{config_key} | {avg_final_acc:.1%}"
        pbar.set_postfix_str(pbarstr)

        if save_results and (config_idx % 5 == 0 or config_idx == (len(configs_averaging) - 1)):
            df = pd.DataFrame(results)
            df.to_csv(fname, index=False)

    return results
