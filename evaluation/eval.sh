# art-fid
cd projects/StyTR-2/evaluation;

CUDA_VISIBLE_DEVICES=0
python eval_artfid.py --sty PATH_FOR_STY_EVAL --cnt PATH_FOR_CNT_EVAL --tar PATH_FOR_STYLIZED_IMAGES
