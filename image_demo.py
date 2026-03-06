import os
from argparse import ArgumentParser

from PIL import Image

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2
import numpy as np
from Mytest import pre_slide, VisualizeSegmm

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/data/fywdata/LoveDA/Test/Urban/images_png', help='Image file')
    parser.add_argument('--config',default='experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_U2R.py',help='test config file path')
    parser.add_argument('--checkpoint',default='/data/fywdata/code/fyw/UDA/APANet/experiments/segformerb5/results_LoveDA/iter_35000.pth',help='checkpoint file')
    parser.add_argument('--save_path',default='/data/fywdata/code/fyw/UDA/APANet/experiments/segformerb5/results_LoveDA/iter_35000',help='checkpoint file')
    parser.add_argument('--device', default='cuda:2', help='Device used for inference')
    parser.add_argument('--palette',default='loveda',help='Color palette used for segmentation map')
    parser.add_argument('--opacity',type=float,default=0.5,help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # 遍历文件夹中的所有图像
    for filename in os.listdir(args.img):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据实际情况添加或修改文件类型
            img_path = os.path.join(args.img, filename)
            # test a single image
            result = inference_segmentor(model, img_path)
            cv2.imwrite(args.save_path + '/' + filename, result[0].astype(np.uint8))
            #print(result[0].shape)
            # vis gt
            '''
            gt = cv2.imread("2_13_3584_2560_4096_3072_gt.png")
            result = []
            result.append(gt[:, :, 0]-1)
            result = tuple(result)
            print(np.unique(result[0]))
            '''

            # show the results
            '''show_result_pyplot(
                model,
                args.img,
                result,
                get_palette(args.palette),
                opacity=args.opacity)'''


if __name__ == '__main__':
    main()

