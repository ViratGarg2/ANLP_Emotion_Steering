import argparse


def str2bool(value):
    """
    Convert a command line argument to a boolean value.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif value.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def comma_separated_ints(value):
    try:
        return [int(x.strip()) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list of integers: {value}")


def comma_separated_floats(value):
    try:
        return [float(x.strip()) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list of integers: {value}")


#: Every runnable experiment type, in the order they are executed when several
#: are requested at once. Baseline first so its floor is logged before the
#: steered numbers; cheap methods before expensive ones.
ALL_EXPERIMENT_TYPES = [
    "baseline",
    "addition",
    "addition-mean",
    "conceptor-paper",
    "conceptor",
    "conceptor-mean",
]


def comma_separated_experiment_types(value):
    """
    Parse one or more experiment types.

    Accepts a single name ("conceptor"), a comma-separated list
    ("addition,conceptor,conceptor-mean"), or "all". Duplicates are dropped and
    the result is ordered by ALL_EXPERIMENT_TYPES, not by the order typed.
    """
    if value.strip().lower() == "all":
        return list(ALL_EXPERIMENT_TYPES)

    requested = [x.strip() for x in value.split(",") if x.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("No experiment type given")

    unknown = [x for x in requested if x not in ALL_EXPERIMENT_TYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown experiment type(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(ALL_EXPERIMENT_TYPES)}, or 'all'"
        )

    return [t for t in ALL_EXPERIMENT_TYPES if t in requested]


def parse_arguments():
    """
    Parse arguments from the command line. For instructions, please run
    `python main.py -h`.
    """
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    # experiment type
    ####################
    parser.add_argument(
        "--experiment_type",
        type=comma_separated_experiment_types,
        default="baseline",
        help="Experiment type(s) to run, comma-separated. One of "
             "baseline, conceptor, conceptor-mean, addition, addition-mean; "
             "or 'all' for every method. Running several in one invocation "
             "shares a single model load, a single activation extraction, and "
             "crucially the SAME sampled prompts, so the results are directly "
             "comparable (default: %(default)s)",
    )
    parser.add_argument(
        "--mc_fit_centred",
        type=str2bool,
        default=False,
        help="For conceptor-mean only: whether mu is ALSO used when fitting C. "
             "false (default) fits C on raw X, so it is identical to the plain "
             "conceptor's C and the only difference is the bias-recovery term "
             "in the injection — the clean one-variable comparison, and what "
             "the derivation describes. true additionally fits C on (X - mu), "
             "changing two things at once; those results are tagged '_fitmc' "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for prompt sampling. Set this to make a run reproducible "
             "and to pair separate invocations on identical prompts "
             "(default: %(default)s — a fresh random draw each run)",
    )

    # task
    ####################
    parser.add_argument(
        "--task", 
        type=str, 
        default="antonyms", 
        # choices=["antonyms", "synonyms", "analogies", "other_tasks"],
        # help="Task for steering (choices: %(choices)s, default: %(default)s)",
        help="""Task for steering (default: %(default)s), can also be a combination of two tasks \
        by separating them with an ampersand, e.g. 'singular-plural&capitalize'""",
    )

    # model
    ####################
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="EleutherAI/gpt-j-6b", 
        help="Name of the model (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float16", 
        choices=["float16", "float32"],
        help="Data type (choices: %(choices)s, default: %(default)s)",
    )

    # dataset details
    ####################
    parser.add_argument(
        "--num_steering_prompts", 
        type=int, 
        default=100, 
        help="Number of samples to steer in total for each config ($N_p$, default: %(default)s)"
    )
    parser.add_argument(
        "--num_steering_examples_per_prompt",
        type=int,
        default=10,
        help="Number of steering examples in each line per prompt ($N$, default: %(default)s)",
    )
    parser.add_argument(
        "--num_input_prompts", 
        type=int, 
        default=1000, 
        help="Number of input prompts to test the model on ($N_t$, default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for steering (default: %(default)s)",
    )

    # steering hyperparameters
    ####################
    parser.add_argument(
        "--list_extraction_layers",
        # nargs="+",
        # type=int,
        type=comma_separated_ints,
        default=list(range(9, 17)),
        help="Layers to steer (can only do 9-17 for GPT-J, default: %(default)s)",
    )
    parser.add_argument(
        "--list_beta_averaging",
        # nargs="+",
        # type=float,
        type=comma_separated_floats,
        default=[2.3],
        help="Beta value(s) for averaging (default: %(default)s)",
    )
    parser.add_argument(
        "--list_beta_conceptor",
        # nargs="+",
        # type=float,
        type=comma_separated_floats,
        default=[3.9],
        help="Beta value(s) for conceptor steering (default: %(default)s)",
    )
    parser.add_argument(
        "--list_apertures_normal",
        # nargs="+",
        # type=float,
        type=comma_separated_floats,
        default=[0.0125],
        help="Aperture value(s) for normal conceptors (default: %(default)s)",
    )
    parser.add_argument(
        "--list_apertures_mean_centered",
        # nargs="+",
        # type=float,
        type=comma_separated_floats,
        default=[0.05],
        help="Aperture value(s) for mean-centered conceptors (default: %(default)s)",
    )
    parser.add_argument(
        "--list_lambda_conceptor",
        type=comma_separated_floats,
        default=[0.5],
        help="Lambda value(s) for conceptor steering strength: "
             "h' = h + lambda * (beta * C - I) h (default: %(default)s)",
    )

    # directories
    ####################
    parser.add_argument(
        "--mean_activations_path",
        type=str,
        default="../../Results/activations_mean_train.pkl",
        help="Path to mean activations (default: %(default)s)",
    )
    parser.add_argument(
        "--results_path", 
        type=str, 
        default="../../Results/", 
        help="Path to save results (default: %(default)s)",
    )
    
    # experiment details
    ####################
    parser.add_argument(
        "--num_experiments", 
        type=int, 
        default=5, 
        help="Number of experiments (default: %(default)s)",
    )
    parser.add_argument(
        "--save_results", 
        type=str2bool,
        default=True,
        help="Flag to save results (default: %(default)s)",
    )

    # config file
    ####################
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the YAML config file (will overwrite all other options)",
    )

    return parser.parse_args()
