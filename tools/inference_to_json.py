import os
from tqdm import tqdm
from mmseg.apis.inference import inference_model, init_model
from tools.inference import make_dir_if_not_exists
from argparse import ArgumentParser

from tools.utils_json_to_json import seg_results_to_json


def parse_args():
    parser = ArgumentParser(description="Model inferencing.")
    parser.add_argument("-cfg", "--config", required=True, help="Config file")
    parser.add_argument("-chkp", "--checkpoint", required=True, help="Checkpoint file")
    parser.add_argument("-inp", "--input_folder", required=True, help="Image folder")
    parser.add_argument("-out", "--out_dir", required=True, help="Output directory")
    parser.add_argument("--gpu", help="GPU device", default=0)
    return parser.parse_args()


def main(args):
    make_dir_if_not_exists(args.out_dir)
    model = init_model(
        config=args.config, checkpoint=args.checkpoint, device=f"cuda:{args.gpu}"
    )
    results = []
    for img in tqdm(os.listdir(args.input_folder)):
        if not img.endswith(".jpg") | img.endswith(".png"):
            continue
        result = inference_model(
            model, f"{args.input_folder}/{img}"
        )  # here you can either specify a path to the image or a numpy array
        results.append(result)

    seg_results_to_json(results, save_path=f"{args.out_dir}/results.json")


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
