#!/bin/bash

helpFunction()
{
	echo ""
	echo "Usage: $0 -p mp4_path"
	echo -e "\t-p path to mp4 files"
	exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
	case "$opt" in
	p ) p="$OPTARG" ;;
	esac
done

# print helpFunction in case parameters are empty
if [ -z "$p" ]
then
	echo "Did not provide a path with mp4 videos";
	helpFunction
fi

MMDET_CONFIG_FILE="~/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py"
MMDET_CHECKPOINT_FILE="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
MMPOSE_CONFIG_FILE="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py"
MMPOSE_CHECKPOINT_FILE="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth"

echo "python mmpose/demo/top_down_video_demo_with_mmdet.py ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} --video-path <> --device cuda"

# cd into mmpose
cd ..

for filename in $p*.mp4
do
	echo ${filename}
	python demo/top_down_video_demo_with_mmdet.py ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} --video-path ${filename} --device cuda
done
