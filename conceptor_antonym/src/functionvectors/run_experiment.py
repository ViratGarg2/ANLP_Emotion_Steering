import os
import torch
from transformer_lens import HookedTransformer

from utils.actadd import cache_activation_vectors
from utils.arguments import parse_arguments
from utils.conceptors import cache_conceptors
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    ########################################################
    # Parse arguments from command line
    ########################################################

    args = parse_arguments()

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
    STEERING_PROMPTS_PATH = f"../../data/functionvectors/{args.task}.json"
    experiment_type = ExperimentType(args.experiment_type)
    BATCH_SIZE = args.batch_size

    # (below is not needed for baseline)
    MEAN_ACTIVATIONS_PATH = args.mean_activations_path
    list_extraction_layers = args.list_extraction_layers
    list_beta_averaging = args.list_beta_averaging
    list_beta_conceptor = args.list_beta_conceptor
    list_apertures_normal = args.list_apertures_normal
    list_apertures_mean_centered = args.list_apertures_mean_centered
    list_lambda_conceptor = args.list_lambda_conceptor

    if not SAVE_RESULTS:
        print("[WARNING] Results will not be saved.")

    if not os.path.exists(STEERING_PROMPTS_PATH):
        raise FileNotFoundError(f"Steering prompts file not found: {STEERING_PROMPTS_PATH}")

    ########################################################
    # Experiment prep
    ########################################################

    if experiment_type in (ExperimentType.ADDITION_MEAN, ExperimentType.CONCEPTOR_MEAN):
        print("*] Loading mean activations")
        mean_train_activations = load_mean_activations(MEAN_ACTIVATIONS_PATH, DEVICE)

    if os.path.exists("../../hf_token.txt"):
        with open("../../hf_token.txt", "r") as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    print("*] Loading model")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE
    )
    model.eval()

    print("*] Generating experiment data")
    experiment_data = generate_experiment_data(
        model=model,
        num_experiments=NUM_EXPERIMENTS,
        steering_prompts_path=STEERING_PROMPTS_PATH,
        num_steering_prompts=NUM_STEERING_PROMPTS,
        num_steering_examples_per_prompt=NUM_STEERING_EXAMPLES_PER_PROMPT,
        num_input_prompts=NUM_INPUT_PROMPTS,
    )

    activations_cache = None
    if experiment_type != ExperimentType.BASELINE:
        print("*] Generating activations cache")
        activations_cache = generate_activations(
            model=model,
            experiment_data=experiment_data,
            list_extraction_layers=list_extraction_layers,
            num_experiments=NUM_EXPERIMENTS,
            device=DEVICE,
        )

    ########################################################
    # Run experiments
    ########################################################

    if experiment_type == ExperimentType.BASELINE:
        run_baseline_experiments(
            model=model,
            experiment_data=experiment_data,
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
                lambda_=lambda_,
            )
            for extraction_layer in list_extraction_layers
            for beta in list_beta_conceptor
            for aperture in list_apertures_normal
            for lambda_ in list_lambda_conceptor
        ]

        conceptors_cache = cache_conceptors(
            activations_cache=activations_cache,
            list_extraction_layers=list_extraction_layers,
            list_apertures=list_apertures_normal,
            num_experiments=NUM_EXPERIMENTS,
        )

        results_conceptoring = run_all_conceptor_experiments(
            model=model,
            experiment_data=experiment_data,
            conceptors_cache=conceptors_cache,
            configs_conceptoring=configs_conceptoring,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=BATCH_SIZE,
        )

    elif experiment_type == ExperimentType.CONCEPTOR_MEAN:
        configs_conceptoring_mean_centered = [
            ExperimentConfigConceptor(
                extraction_layer=extraction_layer,
                beta=beta,
                aperture=aperture,
                lambda_=lambda_,
            )
            for extraction_layer in list_extraction_layers
            for beta in list_beta_conceptor
            for aperture in list_apertures_mean_centered
            for lambda_ in list_lambda_conceptor
        ]

        conceptors_cache = cache_conceptors(
            activations_cache=activations_cache,
            list_extraction_layers=list_extraction_layers,
            list_apertures=list_apertures_mean_centered,
            num_experiments=NUM_EXPERIMENTS,
            mean_train_activations=mean_train_activations,
        )

        results_conceptoring = run_all_conceptor_experiments(
            model=model,
            experiment_data=experiment_data,
            conceptors_cache=conceptors_cache,
            configs_conceptoring=configs_conceptoring_mean_centered,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=BATCH_SIZE,
            mean_train_activations=mean_train_activations,
        )

    elif experiment_type in (ExperimentType.ADDITION, ExperimentType.ADDITION_MEAN):
        configs_averaging = [
            ExperimentConfigAveraging(
                extraction_layer=extraction_layer,
                beta=beta,
            )
            for extraction_layer in list_extraction_layers
            for beta in list_beta_averaging
        ]

        averaged_activations_cache = cache_activation_vectors(
            activations_cache=activations_cache,
            list_extraction_layers=list_extraction_layers,
            num_experiments=NUM_EXPERIMENTS,
        )

        batch_size = 150 if experiment_type == ExperimentType.ADDITION else 50
        mta_arg = None if experiment_type == ExperimentType.ADDITION else mean_train_activations
        results_averaging = run_all_actadd_experiments(
            model=model,
            experiment_data=experiment_data,
            averaged_activations_cache=averaged_activations_cache,
            configs_averaging=configs_averaging,
            save_results=SAVE_RESULTS,
            num_experiments=NUM_EXPERIMENTS,
            results_path=RESULTS_PATH,
            task=TASK,
            batch_size=BATCH_SIZE,
            mean_train_activations=mta_arg,
        )

    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")
