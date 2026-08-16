import os
import random
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
    # args.experiment_type is a LIST — one or several methods to run in this
    # invocation, already de-duplicated and ordered by arguments.py
    experiment_types = [ExperimentType(t) for t in args.experiment_type]
    BATCH_SIZE = args.batch_size

    # Seeding matters here: generate_experiment_data samples prompts with
    # `random`, so without a seed every invocation evaluates on a different
    # draw and separate runs are not comparable. Running several methods in
    # one invocation sidesteps that entirely — they share one draw.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"*] Seeded prompt sampling with {args.seed}")

    # (below is not needed for baseline)
    MEAN_ACTIVATIONS_PATH = args.mean_activations_path
    MC_FIT_CENTRED = args.mc_fit_centred
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

    needs_mean = any(
        t in (ExperimentType.ADDITION_MEAN, ExperimentType.CONCEPTOR_MEAN)
        for t in experiment_types
    )
    mean_train_activations = None
    if needs_mean:
        print("*] Loading mean activations")
        if not os.path.exists(MEAN_ACTIVATIONS_PATH):
            raise FileNotFoundError(
                f"Mean activations not found: {MEAN_ACTIVATIONS_PATH}\n"
                f"  Required by: "
                f"{', '.join(t.value for t in experiment_types if t in (ExperimentType.ADDITION_MEAN, ExperimentType.CONCEPTOR_MEAN))}\n"
                f"  Generate one for this model with mean_activation.py, then "
                f"point --mean_activations_path at it."
            )
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

    # Extracted once and shared by every steering method in this invocation:
    # all of them derive their artifact from the same last-token activations.
    activations_cache = None
    if any(t != ExperimentType.BASELINE for t in experiment_types):
        print("*] Generating activations cache")
        activations_cache = generate_activations(
            model=model,
            experiment_data=experiment_data,
            list_extraction_layers=list_extraction_layers,
            num_experiments=NUM_EXPERIMENTS,
            device=DEVICE,
        )

    # Built lazily and reused: addition and addition-mean share one averaged
    # cache, so it is only computed once even when both are requested.
    averaged_activations_cache = None

    ########################################################
    # Run experiments
    ########################################################

    print(f"\n*] Running {len(experiment_types)} method(s): "
          f"{', '.join(t.value for t in experiment_types)}")

    for run_idx, experiment_type in enumerate(experiment_types, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{run_idx}/{len(experiment_types)}] {experiment_type.value.upper()} — task '{TASK}'")
        print(f"{'=' * 70}")

        if experiment_type == ExperimentType.BASELINE:
            run_baseline_experiments(
                model=model,
                experiment_data=experiment_data,
                num_experiments=NUM_EXPERIMENTS,
                task=TASK,
                results_path=RESULTS_PATH,
                save_results=SAVE_RESULTS,
            )

        elif experiment_type in (ExperimentType.CONCEPTOR,
                                 ExperimentType.CONCEPTOR_PAPER,
                                 ExperimentType.CONCEPTOR_MEAN):
            # The three conceptor variants differ only in which aperture list
            # they sweep, whether mu is threaded through, and whether the pure
            # published form is used. Sharing the body keeps them from
            # drifting apart.
            is_mean = experiment_type == ExperimentType.CONCEPTOR_MEAN
            is_pure = experiment_type == ExperimentType.CONCEPTOR_PAPER
            apertures = list_apertures_mean_centered if is_mean else list_apertures_normal

            # mu enters in TWO independent places for conceptor-mean:
            #   mta_arg  -> the INJECTION, h' = h + lambda*(beta*C - I)(h - mu)
            #   mta_fit  -> the FITTING, C built from (X - mu) instead of X
            # By default only the injection uses mu, so C is byte-identical to
            # the plain conceptor's and conceptor-mean differs from it by the
            # bias-recovery term alone. --mc_fit_centred true also centres the
            # fitting data, which changes two things at once.
            mta_arg = mean_train_activations if is_mean else None
            mta_fit = mta_arg if MC_FIT_CENTRED else None
            if is_mean:
                print("*] conceptor-mean: C fitted on "
                      + ("mean-centred (X - mu) — two changes vs conceptor"
                         if MC_FIT_CENTRED else
                         "raw X — same C as conceptor, bias term isolated"))

            # h' = beta*C*h has no lambda, so sweeping it would just repeat the
            # same experiment N times. Pin it to 1.0, the value at which the
            # residual form reduces to the published one.
            lambdas = [1.0] if is_pure else list_lambda_conceptor
            if is_pure and len(list_lambda_conceptor) > 1:
                print("*] conceptor-paper ignores lambda — pinned to 1.0")

            configs_conceptoring = [
                ExperimentConfigConceptor(
                    extraction_layer=extraction_layer,
                    beta=beta,
                    aperture=aperture,
                    lambda_=lambda_,
                )
                for extraction_layer in list_extraction_layers
                for beta in list_beta_conceptor
                for aperture in apertures
                for lambda_ in lambdas
            ]

            conceptors_cache = cache_conceptors(
                activations_cache=activations_cache,
                list_extraction_layers=list_extraction_layers,
                list_apertures=apertures,
                num_experiments=NUM_EXPERIMENTS,
                mean_train_activations=mta_fit,
            )

            run_all_conceptor_experiments(
                model=model,
                experiment_data=experiment_data,
                conceptors_cache=conceptors_cache,
                configs_conceptoring=configs_conceptoring,
                save_results=SAVE_RESULTS,
                num_experiments=NUM_EXPERIMENTS,
                results_path=RESULTS_PATH,
                task=TASK,
                batch_size=BATCH_SIZE,
                mean_train_activations=mta_arg,
                pure=is_pure,
                # the default coupling keeps the plain name; the two-changes-
                # at-once variant is tagged so they cannot be confused on disk
                fname_suffix="fitmc" if (is_mean and MC_FIT_CENTRED) else "",
            )

            # conceptors are large (d x d per layer/aperture); drop before the
            # next method so several sweeps in one run do not stack up
            del conceptors_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif experiment_type in (ExperimentType.ADDITION, ExperimentType.ADDITION_MEAN):
            configs_averaging = [
                ExperimentConfigAveraging(
                    extraction_layer=extraction_layer,
                    beta=beta,
                )
                for extraction_layer in list_extraction_layers
                for beta in list_beta_averaging
            ]

            # addition and addition-mean use the same averaged vectors, so
            # this is computed on first use and reused afterwards
            if averaged_activations_cache is None:
                averaged_activations_cache = cache_activation_vectors(
                    activations_cache=activations_cache,
                    list_extraction_layers=list_extraction_layers,
                    num_experiments=NUM_EXPERIMENTS,
                )

            mta_arg = None if experiment_type == ExperimentType.ADDITION else mean_train_activations
            run_all_actadd_experiments(
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

    print(f"\n{'=' * 70}")
    print(f"*] Done — {len(experiment_types)} method(s) for task '{TASK}', "
          f"results in {RESULTS_PATH}")
    print(f"{'=' * 70}")
