
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.configuration import default_num_processes

from nnunetv2.evaluation.evaluate_calibration import compute_probabilistic_metrics_on_folder2
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
from nnunetv2.evaluation.evaluate_uncertainty import compute_uncertainty_metrics_on_folder2
def evaluate_all_metrics_folder_entry_point():

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty maps produced by nnUNetv2 inference"
    )
    parser.add_argument(
        "gt_folder", type=str,
        help="Folder with ground truth segmentations"
    )
    parser.add_argument(
        "pred_folder", type=str,
        help="Folder with predicted segmentations (must contain uncertainty_maps/)"
    )
    parser.add_argument('prob_folder', type=str, help='folder with softmax probabilities (for calibration metrics)')
    parser.add_argument(
        "-djfile", type=str, required=True,
        help="dataset.json file"
    )
    parser.add_argument(
        "-pfile", type=str, required=True,
        help="plans.json file"
    )
    parser.add_argument(
        "-output_folder", type=str, required=False,
        default=None,
        help="Output directory. Optional. Default: pred_folder"
    )
    parser.add_argument(
        "-np", type=int, required=False,
        default=default_num_processes,
        help=f"Number of processes used. Optional. Default: {default_num_processes}"
    )
    parser.add_argument('-nbins', type=int, required=False, default=15, help='Number of confidence bins for ECE calculation. Default: 15') # added
    parser.add_argument('-eps', type=float, required=False, default=1e-8, help='Epsilon for NLL calculation. Default: 1e-8') # added
    parser.add_argument(
        "--chill", action="store_true",
        help="Do not crash if uncertainty maps are missing for some cases"
    )

    args = parser.parse_args()


    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, join(args.output_folder, 'segmentation_metrics.json'), args.np, chill=args.chill)


    compute_uncertainty_metrics_on_folder2(
        folder_ref=args.gt_folder,
        folder_pred=args.pred_folder,
        dataset_json_file=args.djfile,
        plans_file=args.pfile,
        output_dir=args.output_folder,
        num_processes=args.np,
        chill=args.chill,
    )

    if args.prob_folder is not None:
        compute_probabilistic_metrics_on_folder2(
            folder_ref=args.gt_folder,
            folder_prob=args.prob_folder,
            dataset_json_file=args.djfile,
            plans_file=args.pfile,
            output_file=join(args.output_folder, 'calibration_metrics.json'),
            n_bins=args.nbins,
            eps=args.eps,
            num_processes=args.np,
            chill=args.chill,
        )
