import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import dill
from sklearn import preprocessing

import sys

sys.path.append("_dat")
sys.path.append("_src")

# Proposed method
import CubeScope


def prepare_event_tensor(
    given_data,
    categorical_idxs,
    time_idx,
    freq,
    return_encoders=False,
    save_encoders=False,
    outdir="./",
):
    data = given_data.copy("deep")
    print(data)
    print(data.columns)
    data = data.dropna(subset=(categorical_idxs + [time_idx]))

    # Encode timestamps
    data[time_idx] = data[time_idx].round(freq)
    data = data.sort_values(time_idx)
    start = data[time_idx].min()
    end = data[time_idx].max()
    ticks = pd.date_range(start, end, freq=freq)
    timepoint_encoder = preprocessing.LabelEncoder()
    timepoint_encoder.fit(ticks)
    data[time_idx] = timepoint_encoder.transform(data[time_idx].values)

    # Encode categorical features
    oe = preprocessing.OrdinalEncoder()
    data[categorical_idxs] = oe.fit_transform(data[categorical_idxs])
    data[categorical_idxs] = data[categorical_idxs].astype(int)

    if save_encoders:
        # Timestamps
        time_encoder = pd.DataFrame(
            timepoint_encoder.classes_,
            index=range(len(timepoint_encoder.classes_)),
            columns=["timestamp"],
        )

        time_encoder.to_csv(outdir + f"/{time_idx}.csv.gz", index=False)

        # Categorical features
        for key, feature_elem in zip(categorical_idxs, oe.categories_):
            ctg_encoder = pd.DataFrame(
                feature_elem, index=range(len(feature_elem)), columns=[key]
            )
            ctg_encoder.to_csv(outdir + "/" + key + ".csv.gz", index=False)

    if return_encoders:
        return data.reset_index(drop=True), oe, timepoint_encoder
    else:
        return data.reset_index(drop=True)


def plot_ground_truth(ax, pattern, width=100):
    if pattern=="121":
        width=60
    pattern_idxs = list(pattern)
    pattern_idxs = [int(s) for s in pattern_idxs]
    st = 0
    for pattern_id in pattern_idxs:
        ax.hlines(y=pattern_id, xmin=st, xmax=st + width, color="gray", linewidth=8)
        st += width
    # ax.set_yticklabels([int(i) for i in ax.get_yticks()])
    ax.set_title("Ground truth")


def plot_regimeassignment(
    ax,
    CubeScope: object,
    regime_assignments: list,
    skip=[],
    line_width=8,
    rotation=60,
    with_plot_ax=None,
    show_label_type="",
    time_ind=0,
    # replace_time=False,
    # replace_time_width=500,
):
    all_rgm_num = len(CubeScope.regimes)
    length = CubeScope.data_len
    ax.set_title("Regime assignments")
    r = regime_assignments[0][0]
    st = regime_assignments[0][1]
    for assign in regime_assignments[1:]:
        ed = assign[1]
        if r not in skip:
            ax.hlines(
                y=r - len(skip),
                xmin=st,
                xmax=ed,
                color=sns.color_palette(n_colors=all_rgm_num)[r - len(skip)],
                linewidth=line_width,
            )
            if with_plot_ax is not None:
                with_plot_ax.axvline(x=st, c="gray", linestyle="-", linewidth=1.2)
        r = assign[0]
        st = assign[1]
    ed = length
    ax.hlines(
        y=r - len(skip),
        xmin=st,
        xmax=ed,
        color=sns.color_palette(n_colors=all_rgm_num)[r - len(skip)],
        linewidth=line_width,
    )
    if with_plot_ax is not None:
        with_plot_ax.axvline(x=st, c="gray", linestyle="-", linewidth=1.2)
    ax.set_yticks(range(all_rgm_num - len(skip)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument("--input_fpath", type=str)  #
    parser.add_argument("--out_dir", type=str)  #
    parser.add_argument("--categorical_idxs", type=str)
    parser.add_argument("--time_idx", type=str)
    parser.add_argument("--freq", type=str, default="H")
    parser.add_argument("--init_len", type=int)

    # model
    parser.add_argument("--k", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--label_col", type=str, default="None")
    

    # for experiments
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pattern", type=str) # for synthetic data

    # model details
    parser.add_argument("--maxl", type=int, default=5)
    parser.add_argument("--N_ITER", type=int, default=20)

    args = parser.parse_args()

    # make output dir
    outputdir = args.out_dir
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    # data I/O
    categorical_idxs = args.categorical_idxs.split("/")
    time_idx = args.time_idx

    raw_df = pd.read_csv(args.input_fpath)
    raw_df[time_idx] = pd.to_datetime(raw_df[time_idx])

    if args.anomaly and args.label_col!="None":
        anom_series = raw_df[args.label_col]
    else:
        anom_series = np.zeros(len(raw_df))

    # last column: timestamp
    tensor = prepare_event_tensor(
        raw_df,
        categorical_idxs,
        time_idx,
        freq=args.freq,
        outdir=outputdir,
    )

    tensor = tensor[[time_idx] + categorical_idxs]

    tensor_shape = tensor.max().values + 1

    print(f"--Input dataset--")
    print(tensor)
    tensor_shape = (tensor.max().values + 1).astype("int64")
    n_full_cells = len(tensor.groupby(categorical_idxs + [time_idx]).count())

    print(f"--Dataset description--")
    print(f"tensor shape: {tensor_shape}")
    print(f"# of records: {len(tensor)}")
    print(f"sparsity(%): {1 - n_full_cells / np.prod(tensor_shape)}")
    print(f"------------------------")

    # tau
    width = int(args.width)
    init_len = int(args.init_len)

    # fixed hyperparameter
    args.alpha = 1 / args.k
    args.beta = 1 / args.k

    # Set inlier tensor to train tensor
    assert len(tensor) == len(anom_series)

    tensor_Train = (
        tensor.loc[(tensor[args.time_idx] < init_len) & (anom_series == 0), :]
        .copy("deep")
        .reset_index(drop=True)
    )

    # align time_idx
    tensor_Train[args.time_idx] = (
        tensor_Train[args.time_idx]
        .rank(method="dense", numeric_only=True)
        .astype("int64")
        - 1
    )
    print(f"train tensor shape: {tensor_Train.max().values+1}")

    # train
    CubeScope = CubeScope.CubeScope(
        tensor,
        args.k,
        width,
        init_len,
        outputdir,
        args=args,
        verbose=args.verbose,
        tensor_shape=tensor_shape,
    )

    ### Batch processing (Initialize) ###################
    start_time = time.process_time()
    regime_assignments = CubeScope.init_infer(tensor_Train, n_iter=args.N_ITER)
    elapsed_time = time.process_time() - start_time
    print(f"Elapsed time(train): {elapsed_time:.2f} [sec]")
    if args.verbose:
        outputdir_s = outputdir + "/train/"
        if os.path.exists(outputdir_s):
            shutil.rmtree(outputdir_s)
        os.makedirs(outputdir_s)
        CubeScope.save(outputdir_s)
    ### Batch processing (Initialize) ###################

    ### Stream processing ###############################
    start_time_stream_process = time.process_time()
    times = []
    max_ = tensor[time_idx].max() + 1
    CubeScope.data_len = max_

    for i in range(0, max_, width):
        start_time = time.process_time()
        current_tensor = tensor[
            (tensor[args.time_idx] >= i) & (tensor[args.time_idx] < (i + width))
        ]
        current_tensor.loc[:, args.time_idx] -= i
        shift_id = CubeScope.infer_online(
            current_tensor, args.alpha, args.beta, n_iter=args.N_ITER
        )
        elapsed_time = time.process_time() - start_time
        print(f"Elapsed time(online#{i}): {elapsed_time:.2f} [sec]")
        times.append(elapsed_time)
        if args.verbose:
            outputdir_s = f"{outputdir}/t_{str(i)}/"
            if os.path.exists(outputdir_s):
                shutil.rmtree(outputdir_s)
            os.makedirs(outputdir_s)
            CubeScope.save(outputdir_s)

        if type(shift_id) == int:
            prev_n = i
            regime_assignments.append([shift_id, prev_n])
    CubeScope.rgm_update_fin()
    if args.verbose:
        CubeScope.save(outputdir_s)
    elapsed_time_stream_process = time.process_time() - start_time_stream_process
    print(
        f"Elapsed time(all stream processing): {elapsed_time_stream_process:.2f} [sec]"
    )
    ### Stream processing ###############################

    # save overall results
    print(f"result in {outputdir}")
    result = [CubeScope, regime_assignments, times]
    dill.dump(result, open(f"{outputdir}/result.dill", "wb"))
    with open(outputdir + "/keys.csv", "w") as f:
        f.write(",".join(tensor.columns))

    # viz temporal pattern segmetation
    fig, axes = plt.subplots(2, 1, figsize=(15, 4))
    plot_ground_truth(axes[0], args.pattern)
    print(regime_assignments)

    plot_regimeassignment(axes[1], CubeScope, regime_assignments)
    fig.tight_layout()
    fig.savefig(outputdir + "/segmentation_results.png")
