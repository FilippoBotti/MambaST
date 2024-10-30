
python main.py --model_name mambast --style_dir datasets/wikiart/ --content_dir datasets/train2014  \
                --vgg checkpoints/vgg_normalised.pth --print_every 1000 \
                --checkpoints_dir checkpoints/ --results_dir outputs/  \
                --batch_size 8 --mode train --d_state 16  --lr 1e-4 --rnd_style