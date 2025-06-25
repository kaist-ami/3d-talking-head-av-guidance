import argparse
import os
from tqdm import tqdm

from utils.utils import *


def compute_lve(src_dir, gt_dir, vertice_dim, lip_mask, save_fpath):
    file_list = os.listdir(src_dir)

    vertices_gt_all = []
    vertices_pred_all = []

    for name in tqdm(file_list, desc="Computing LVE ..."):
        vertices_pred = np.load(os.path.join(src_dir, name)).reshape(-1, vertice_dim//3, 3) # [T, N, 3]

        if "condition" in name:
            name = "_".join(name.split("_")[:5]) + ".npy"
        vertices_gt = np.load(os.path.join(gt_dir, name))[::2].reshape(-1, vertice_dim//3, 3) # [T, N, 3]

        if len(vertices_pred) > len(vertices_gt):
            vertices_pred = vertices_pred[:vertices_gt.shape[0]]
        elif len(vertices_pred) < len(vertices_gt):
            vertices_gt = vertices_gt[:vertices_pred.shape[0]]
        assert len(vertices_pred) == len(vertices_gt)
        
        vertices_gt_all.extend(list(vertices_gt))
        vertices_pred_all.extend(list(vertices_pred))

    vertices_gt_all = np.array(vertices_gt_all) # (T_all, N, 3)
    vertices_pred_all = np.array(vertices_pred_all)

    L2_dis_mouth_max = np.array(
        [
            np.square(vertices_gt_all[:, v, :] - vertices_pred_all[:, v, :])
            for v
            in lip_mask
        ]
    ) # (N_lip, T_all, 3)
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2)) # (T_all, N_lip, 3)
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2) # (T_all, N_lip)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1) # (T_all, )

    lve = np.mean(L2_dis_mouth_max)
    print("LVE: ", lve)

    with open(save_fpath, "w") as f:
        f.write(f"Source files in: {src_dir} (# files: {len(file_list)})\n")
        f.write(f"LVE: {lve}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Compute LVE"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to test the model",
    )
    parser.add_argument(
        "--save_root_dir", type=str, default="outputs/vocaset",
    )
    cmd_input = parser.parse_args()

    # Load config
    if cmd_input.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif cmd_input.dataset == "BIWI":
        args = load_config("config/biwi.yaml")

    # Set the paths
    if cmd_input.save_root_dir is not None:
        os.makedirs(cmd_input.save_root_dir, exist_ok=True)
        args.save_root_dir = cmd_input.save_root_dir
    elif args.save_root_dir is not None:
        os.makedirs(cmd_input.save_root_dir, exist_ok=True)
    else:
        print("Please specify the `save_root_dir`!")

    args.save_pred_path = os.path.join(args.save_root_dir, args.save_pred_path) # directory
    args.save_result_path = os.path.join(args.save_root_dir, args.save_result_path) # directory
    args.save_result_fname = os.path.join(args.save_result_path, "lve.txt")

    # Make directories to save results
    make_dirs(args.save_result_path)

    # Get lip mask
    lip_mask = get_lip_verts(args.dataset)

    # Compute
    compute_lve(
        src_dir = args.save_pred_path,
        gt_dir = os.path.join(args.dataset, args.vertices_path),
        vertice_dim=args.vertice_dim,
        lip_mask=lip_mask,
        save_fpath = args.save_result_fname
    )