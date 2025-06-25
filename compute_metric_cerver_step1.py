import argparse
import os
import re
import json
from tqdm import tqdm

from utils.utils import *

VOCASET_TEST_ID = ["FaceTalk_170731_00024_TA", "FaceTalk_170809_00138_TA"]
VOCASET_TEST_SENTENCE = range(21, 41)
BIWI_TEST_ID = ["F1", "F5", "F6", "F7", "F8", "M1", "M2", "M6"]
BIWI_TEST_SENTENCE = range(37, 41)


def build_vsr(model_path):
    # Load configurations
    from hydra import compose, initialize
    initialize(version_base="1.3", config_path="../auto_avsr/configs")
    cfg = compose(config_name="config")
    assert os.path.exists(model_path), "Lip reader model is not exist!"
    cfg.pretrained_model_path = model_path
    
    # Load lip reader model
    from auto_avsr.lightning import ModelModule
    cfg.data.modality = "video"
    vsr_model = ModelModule(cfg)
    vsr_model.model.load_state_dict(
        torch.load(
            cfg.pretrained_model_path, map_location=lambda storage,loc:storage,
        )
    )
    print(f"VSR model is loaded - {model_path}")
    return vsr_model

@torch.no_grad()
def get_predicted_text(
    src_dir, vsr_model_path, render_obj_filename, dataset, vertice_dim, 
    save_output_fpath, device="cuda", is_gt=False
):
    assert save_output_fpath.endswith(".json")

    import torchvision
    from utils.renderer_pytorch3d import set_rasterizer, SRenderY

    # Build lip reader model
    vsr_model = build_vsr(vsr_model_path)
    vsr_model.to(device)
    vsr_model.eval()

    # For Render
    set_rasterizer()
    render = SRenderY(
        image_size=224,
        obj_filename=render_obj_filename,
        uv_size=256,
    ).to(device)
    video_transform = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(88),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize(0.421, 0.4),
    )

    # GT text
    sentence_path = os.path.join(args.dataset, args.sentence_path)
    sentences = {}
    if args.dataset == "vocaset":
        for sentence_file in os.listdir(sentence_path):
            if not sentence_file.endswith("txt"):
                continue
            identity = sentence_file[:sentence_file.find('.txt')]
            sentence_file = os.path.join(sentence_path, sentence_file)
            with open(sentence_file) as f:
                lines = f.readlines()
                sentences[identity] = [line.strip() for line in lines if line!="\n"]
    elif args.dataset=="BIWI":
        with open(sentence_path) as f:
            lines = f.readlines()
        for l_idx, l in enumerate(lines):
            sentences[l_idx+1] = l.strip()

    results = []

    file_list = sorted(os.listdir(src_dir))
    for name in tqdm(file_list, desc="Extracting text predictions..."):
        if is_gt and dataset=="vocaset":
            identity = name[:name.find("_sentence")]
            s_idx_end = name.find('.npy')
            sentence_num = int(name[s_idx_end-2:s_idx_end])-1 # list starts from 0
            if identity in VOCASET_TEST_ID and sentence_num in VOCASET_TEST_SENTENCE:
                pass
            else:
                continue
        
        if is_gt and dataset=="BIWI":
            identity = name[:name.find("_e")]
            s_idx_end = name.find('.npy')
            sentence_num = int(name[s_idx_end-2:s_idx_end])-1 # list starts from 0
            if identity in BIWI_TEST_ID and sentence_num in BIWI_TEST_SENTENCE:
                pass
            else:
                continue

        prediction = np.load(os.path.join(src_dir, name)).reshape(-1, vertice_dim//3, 3) # [T, 5023, 3]
        prediction = torch.tensor(prediction, dtype=torch.float32).cuda()

        if is_gt and dataset=="vocaset":
            prediction = prediction[::2]

        if dataset=="vocaset":
            proj_camera = torch.Tensor([8, 0, 0]).expand(len(prediction), -1).cuda()
        elif dataset=="BIWI":
            proj_camera = torch.Tensor([1.6, 0, 0]).expand(len(prediction), -1).cuda()
        trans_verts = batch_orth_proj(prediction, proj_camera) # [frame, num_verts, 3]
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        # Render face
        rendered_video = render.render_shape(prediction, trans_verts) # [frame, C=3, H=224, W=224]

        # Crop lip region
        if dataset=="vocaset":
            rendered_video = rendered_video[:, :, 107:203, 63:159] # [frame, 3, 96, 96]
        elif dataset=="BIWI":
            rendered_video = rendered_video[:, :, 107:203, 68:164] # [frame, 3, 96, 96]

        # Transform rendered video
        video = video_transform(rendered_video) # [frame, 1, 88, 88]

        # Get predicted text
        pred_text = vsr_model(video).lower()

        # Get gt text
        if args.dataset=="vocaset":
            split_idx = name.find("_sentence")
            identity = name[:split_idx]
            sentence_num = int(name[split_idx+9:split_idx+11])-1 # list starts from 0
            gt_text = sentences[identity][sentence_num].lower()
        elif args.dataset=="BIWI":
            s_idx_end = name.find('.npy')
            sentence_num = int(name[s_idx_end-2:s_idx_end])
            gt_text = sentences[sentence_num].lower()

        # Record results
        item = {
            "key": name,
            "pred_text": re.sub(r"[^a-zA-Z]", " ", pred_text).strip(),
            "gt_text": re.sub(r"[^a-zA-Z]", " ", gt_text).strip()
        }
        results.append(item)

    with open(save_output_fpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Get lip reading prediction from VSR model to compure VER/CER"
    )
    parser.add_argument(
        "--dataset", type=str, default="vocaset", help="Dataset to test the model",
    )
    parser.add_argument(
        "--save_root_dir", type=str, default="outputs/vocaset", help="Directory to save result files",
    )
    parser.add_argument(
        "--vsr_model_path", type=str, help="Path of VSR model",
    )
    parser.add_argument(
        "--is_gt", action="store_true", help="If you want to compute for GT vertices",
    )
    cmd_input = parser.parse_args()

    # Load config
    if cmd_input.dataset=="vocaset":
        args = load_config("config/vocaset.yaml")
    elif cmd_input.dataset == "BIWI":
        args = load_config("config/biwi.yaml")
    args.target_metric = cmd_input.metric

    # Set the paths
    if cmd_input.save_root_dir is not None:
        os.makedirs(cmd_input.save_root_dir, exist_ok=True)
        args.save_root_dir = cmd_input.save_root_dir
    elif args.save_root_dir is not None:
        os.makedirs(cmd_input.save_root_dir, exist_ok=True)
    else:
        print("Please specify the `save_root_dir`!")

    if cmd_input.is_gt:
        args.save_pred_path = os.path.join(args.dataset, args.vertices_path)
        print("Pred path: ", args.save_pred_path)
    else:
        args.save_pred_path = os.path.join(args.save_root_dir, args.save_pred_path) # directory
    args.save_result_path = os.path.join(args.save_root_dir, args.save_result_path) # directory
    args.save_result_fname = os.path.join(args.save_result_path, "cer_ver.txt")

    # Make directories to save results
    make_dirs(args.save_result_path)

    pred_text_list = get_predicted_text(
        src_dir=args.save_pred_path,
        vsr_model_path=args.vsr_model_path,
        render_obj_filename=args.obj_filename,
        dataset=args.dataset,
        vertice_dim=args.vertice_dim,
        save_output_fpath=args.save_result_fname.replace(".txt", ".json"),
        device="cuda",
        is_gt=cmd_input.is_gt
    )