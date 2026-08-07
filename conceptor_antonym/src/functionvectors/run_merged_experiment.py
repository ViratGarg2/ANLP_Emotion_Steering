import os
import torch
from transformer_lens import HookedTransformer

from utils.actadd import cache_activation_vectors
from utils.arguments import parse_arguments
from utils.conceptors import cache_conceptors, combine_conceptors_and, combine_conceptors
from utils.configs import ExperimentConfigConceptor, ExperimentType, ExperimentConfigAveraging
from utils.data import generate_experiment_data, load_mean_activations
from experiments import (
    run_baseline_experiments,
    run_all_conceptor_experiments,
    run_all_actadd_experiments,
)
from utils.steering import generate_activations


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    ########################################################
    # Parse arguments from command line
    ########################################################

    args = parse_arguments()
    assert "&" in args.task, "Task must be a combination of two tasks separated by an ampersand"

    # Log the arguments
    print("\n*] Launching experiment with arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    # Assign parsed arguments to variables
    TASK = args.task
    RESULTS_PATH = args.results_path
    MODEL_NAME = args.model_name
    DTYPE = args.dtype
    SAVE_RESULTS = args.save_results
    NUM_STEERING_PROMPTS = args.num_steering_prompts
    NUM_STEERING_EXAMPLES_PER_PROMPT = args.num_steering_examples_per_prompt
    NUM_INPUT_PROMPTS = args.num_input_prompts
    NUM_EXPERIMENTS = args.num_experiments
    STEERING_PROMPTS_PATHS = {
        "combined": f"../../data/functionvectors/{args.task.replace('&', '-')}.json",
        "first": f"../../data/functionvectors/{args.task.split('&')[0]}.json",
        "second": f"../../data/functionvectors/{args.task.split('&')[1]}.json",
    }
    experiment_type = ExperimentType(args.experiment_type)

    # (below is not needed for baseline)
    MEAN_ACTIVATIONS_PATH = args.mean_activations_path
    list_extraction_layers = args.list_extraction_layers
    list_beta_averaging = args.list_beta_averaging
    list_beta_conceptor = args.list_beta_conceptor
    list_apertures_normal = args.list_apertures_normal
    list_apertures_mean_centered = args.list_apertures_mean_centered

    if not SAVE_RESULTS:
        print("[WARNING] Results will not be saved.")

    for STEERING_PROMPTS_PATH in STEERING_PROMPTS_PATHS.values():
        if not os.path.exists(STEERING_PROMPTS_PATH):
            raise FileNotFoundError(f"Steering prompts file not found: {STEERING_PROMPTS_PATH}")

    ########################################################
    # Experiment prep
    ########################################################

    if experiment_type in (ExperimentType.ADDITION_MEAN, ExperimentType.CONCEPTOR_MEAN):
        print("*] Loading mean activations")
        mean_train_activations = load_mean_activations(MEAN_ACTIVATIONS_PATH, DEVICE)

    print("*] Loading model")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE
    )
    model.eval()

    print("*] Generating experiment data")
    experiment_data = {
        key: generate_experiment_data(
            model=model,
            num_experiments=NUM_EXPERIMENTS,
            steering_prompts_path=STEERING_PROMPTS_PATH,
            num_steering_prompts=NUM_STEERING_PROMPTS,
            num_steering_examples_per_prompt=NUM_STEERING_EXAMPLES_PER_PROMPT,
            num_input_prompts=NUM_INPUT_PROMPTS,
        )
        for key, STEERING_PROMPTS_PATH in STEERING_PROMPTS_PATHS.items()
    }

    activations_cache = None
    if experiment_type != ExperimentType.BASELINE:
        print("*] Generating activations cache")
        activations_cache = {
            key: generate_activations(
                model=model,
                experiment_data=experiment_data,
                list_extraction_layers=list_extraction_layers,
                num_experiments=NUM_EXPERIMENTS,
                device=DEVICE,
            )
            for key, experiment_data in experiment_data.items()
        }

    ########################################################
    # Run experiments
    ########################################################

    if experiment_type == ExperimentType.BASELINE:
        run_baseline_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            num_experiments=NUM_EXPERIMENTS,
            task=TASK,
            results_path=RESULTS_PATH,
            save_results=SAVE_RESULTS,
        )

    elif experiment_type == ExperimentType.CONCEPTOR:
        configs_conceptoring = [
            ExperimentConfigConceptor(
                extraction_layer=extraction_layer,
                beta=beta,
                aperture=aperture,
            )
            for extraction_layer in list_extraction_layers
            for beta in list_beta_conceptor
            for aperture in list_apertures_normal
        ]

        conceptors_cache = {}
        for key in activations_cache.keys():
            print(f"*] Cache conceptors for {key}")
            conceptors_cache[key] = cache_conceptors(
                activations_cache=activations_cache[key],
                list_extraction_layers=list_extraction_layers,
                list_apertures=list_apertures_normal,
                num_experiments=NUM_EXPERIMENTS,
            )

        print(f"*] Combining conceptors")
        conceptors_cache["boolean"] = conceptors_cache["first"]
        for exp_idx in conceptors_cache["combined"].keys():
            for layer_aperture in conceptors_cache["combined"][exp_idx].keys():
                conceptors_cache["boolean"][exp_idx][layer_aperture] = combine_conceptors(
                    conceptors_cache["first"][exp_idx][layer_aperture],
                    conceptors_cache["second"][exp_idx][layer_aperture]
                )

        # run it with the real conceptor
        print(f"*] Run boolean conceptor experiments")
        results_conceptoring = run_all_conceptor_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            conceptors_cache=conceptors_cache["boolean"],
            configs_conceptoring=configs_conceptoring,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=50,
            fname_suffix="boolean",
        )

        print(f"*] Combining conceptors")
        conceptors_cache["boolean"] = conceptors_cache["first"]
        for exp_idx in conceptors_cache["combined"].keys():
            for layer_aperture in conceptors_cache["combined"][exp_idx].keys():
                conceptors_cache["boolean"][exp_idx][layer_aperture] = combine_conceptors_and(
                    conceptors_cache["first"][exp_idx][layer_aperture],
                    conceptors_cache["second"][exp_idx][layer_aperture]
                )

        # run it with the real conceptor
        print(f"*] Run boolean conceptor experiments")
        results_conceptoring = run_all_conceptor_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            conceptors_cache=conceptors_cache["boolean"],
            configs_conceptoring=configs_conceptoring,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=50,
            fname_suffix="booleanand",
        )

        # run it with the real conceptor
        print(f"*] Run standard combined conceptor experiments")
        results_conceptoring = run_all_conceptor_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            conceptors_cache=conceptors_cache["combined"],
            configs_conceptoring=configs_conceptoring,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=50,
            fname_suffix="combined",
        )

    elif experiment_type == ExperimentType.CONCEPTOR_MEAN:
        raise NotImplementedError("Mean-centered conceptor experiments are not implemented yet.")

    elif experiment_type == ExperimentType.ADDITION_MEAN:
        raise NotImplementedError("Addition experiments are not implemented yet.")

    elif experiment_type == ExperimentType.ADDITION:
        configs_averaging = [
            ExperimentConfigAveraging(
                extraction_layer=extraction_layer,
                beta=beta,
            )
            for extraction_layer in list_extraction_layers
            for beta in list_beta_averaging
        ]

        averaged_activations_cache = {}
        for key in activations_cache.keys():
            print(f"*] Cache activation vectors for {key}")
            averaged_activations_cache[key] = cache_activation_vectors(
                activations_cache=activations_cache[key],
                list_extraction_layers=list_extraction_layers,
                num_experiments=NUM_EXPERIMENTS,
            )

        batch_size = 150 if experiment_type == ExperimentType.ADDITION else 50
        mta_arg = None if experiment_type == ExperimentType.ADDITION else mean_train_activations

        print("*] Run addition combined experiment with v(1,2)")
        results_averaging = run_all_actadd_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            averaged_activations_cache=averaged_activations_cache["combined"],
            configs_averaging=configs_averaging,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=batch_size,
            mean_train_activations=mta_arg,
            fname_suffix="combined",
        )

        print("*] Merge steering vectors (v(1)+v(2))/2")
        averaged_activations_cache["jointactivations"] = averaged_activations_cache["first"]
        for exp_idx in averaged_activations_cache["combined"].keys():
            for layer in averaged_activations_cache["combined"][exp_idx].keys():
                averaged_activations_cache["jointactivations"][exp_idx][layer] = (
                    averaged_activations_cache["first"][exp_idx][layer] +
                    averaged_activations_cache["second"][exp_idx][layer]
                ) / 2.0
        print("*] Run addition merged experiment")
        results_averaging = run_all_actadd_experiments(
            model=model,
            experiment_data=experiment_data["combined"],
            averaged_activations_cache=averaged_activations_cache["jointactivations"],
            configs_averaging=configs_averaging,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=batch_size,
            mean_train_activations=mta_arg,
            fname_suffix="merged",
        )

    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")
