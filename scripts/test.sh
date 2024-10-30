
python main.py  --mode test --style_dir data/sty --content_dir data/cnt     \
                --vgg checkpoints/vgg_normalised.pth \
                --decoder_path checkpoints/decoder_iter_160000.pth --embedding_path checkpoints/embedding_iter_160000.pth \
                --Trans_path checkpoints/transformer_iter_160000.pth \
                --output_dir results --seed 123456 \
                --rnd_style --img_size 512 --d_state 16 