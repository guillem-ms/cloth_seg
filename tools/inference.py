from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
from argparse import ArgumentParser
from tqdm import tqdm

import os

def make_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        

def parse_args():
    parser = ArgumentParser(description='Model inferencing.')
    parser.add_argument('-cfg', '--config', required=True, help='Config file')
    parser.add_argument('-chkp', '--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('-inp', '--input_folder', required=True, help='Image folder')
    parser.add_argument('-out', '--out_dir', required=True, help='Output directory')
    parser.add_argument('--gpu',  help='GPU device', default=0)
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--gt', action='store_true', help='draw ground truth bounding boxes')
    parser.add_argument('--suffix', type=str, help='Output file suffix', default='inf')
    return parser.parse_args()




def main(args):
    model = init_model(
        config=args.config, 
        checkpoint=args.checkpoint, 
        device=f"cuda:{args.gpu}"
    )
    make_dir_if_not_exists(args.out_dir)
    for img in tqdm(os.listdir(args.input_folder)):
        result = inference_model(model, f"{args.input_folder}/{img}")
        show_result_pyplot(
            model, 
            img=f"{args.input_folder}/{img}", 
            out_file=f"{args.out_dir}/{img[:-4]}_{args.suffix}.png",
            draw_gt=False, 
            show=args.show, 
            result=result, 
            save_dir=args.out_dir
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)