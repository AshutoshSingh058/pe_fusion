"""Late (average) fusion script.

Combines prediction probability files from independently trained models by
averaging their probability scores.  This is the simplest late-fusion
strategy and corresponds to the *late elastic average* configuration
described in the paper.

Typical usage::

    python models/late_fusion.py \\
        --result_paths results/emr_model/results.csv,results/imaging_model/results.csv \\
        --late_fusion_name late_all
"""

import argparse
import pandas as pd
import os 
import sys
sys.path.append(os.getcwd())

from constants import *


def main(args):
    """Run late average fusion over the provided result files.

    Reads each CSV result file (which must contain at minimum *accession*,
    *labels*, and *probs* columns), joins them on the accession number, and
    writes a new CSV with the averaged probability to
    ``RESULTS_DIR / late_fusion_name / results.csv``.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.  Required
            attributes: ``result_paths`` (list of str), ``late_fusion_name``
            (str).
    """

    # read in all results
    dfs = []
    for path in args.result_paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    # join results df
    joint_df = dfs[0]
    joint_df = joint_df.set_index(ACCESSION_COL)
    for idx, df in enumerate(dfs[1:]):
        df = df[[ACCESSION_COL, PROBS_COL]]
        df = df.set_index(ACCESSION_COL)
        joint_df = joint_df.join(df, how='left', rsuffix=str(idx))
    joint_df = joint_df.reset_index().rename({'index': ACCESSION_COL})

    # get probability columns
    prob_cols = [c for c in joint_df.columns if 'probs' in c]
    
    # aggregate probabilities
    agg = lambda x: x[prob_cols].mean()
    joint_df[PROBS_COL] = joint_df.apply(agg, axis=1)

    # save results 
    results_dir = RESULTS_DIR / args.late_fusion_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir/ 'results.csv'
    joint_df = joint_df[[ACCESSION_COL, LABELS_COL, PROBS_COL]]
    joint_df.to_csv(results_path)
    print(f'Results saved at {results_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_paths', 
        type=str, 
        help='Paths to the results files, seperated by comma',
    )
    parser.add_argument(
        '--late_fusion_name', 
        type=str, 
        help='Name of the late fusion type',
        default='late_all'
    )
    args = parser.parse_args()

    # create list of result paths 
    args.result_paths = args.result_paths.split(',')

    main(args)