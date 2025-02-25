# Mamba-ST: State Space Model for Efficient Style Transfer (WACV 2025)
*Authors: Filippo Botti, Alex Ergasti, Leonardo Rossi, Tomaso Fontanini, Claudio Ferrari, Massimo Bertozzi and Andrea Prati*

This repository is the official implementation of [Mamba-ST: State Space Model for Efficient Style Transfer](https://www.arxiv.org/abs/2409.10385).

This paper explores a novel design of Mamba, called Mamba-ST, to perform style transfer.

## Results presentation 
<p align="center">
<img src="https://github.com/FilippoBotti/MambaST/blob/main/Figure/generated_images.jpg" width="90%" height="90%">
</p>
Examples of generated images from our Mamba model given a style and a content image. <br>


## Framework
<p align="center">
<img src="https://github.com/FilippoBotti/MambaST/blob/main/Figure/Mamba-Arch.png" width="100%" height="100%">
</p> 
a) Mamba-ST full architecture. It takes as input a content and a style image and generates the content image stylized as the style image. b) Mamba encoder with an additional skip connection (rightmost). c) Our Mamba-ST Decoder, which takes both style and content as input. In particular, style embeddings are shuffled before passing to ST-VSSM in order to loose spatial information, maintaining only higher level information. d) The inner architecture of the Base VSSM. e) The inner architecture of the Base 2D-SSM. f) Our ST-VSSM. Notably, DWConv is shared among content and style embedding. g) Our modified ST 2D-SSM, where the matrices A, B and Delta are computed from the style, the input of the selective scan are the style embedding and the matrix C is calculated using the content.

## Experiment
### Requirements
In order to run the project please install the environment by following these commands: 
```
conda create -n mambast
pip install -r requirements.txt
conda activate mambast
```

You can find the random images used in order to generated the results inside ./data folder.
Please modify all the .sh files with the correct path for your checkpoints and images before 
running the following instructions.

### Evaluation 
[Pretrained models] (https://drive.google.com/drive/folders/1pVhJFwk2f3arP7zUDFAe5_PJrPSG1gc2?usp=drive_link) <br> 
```
sh scripts/eval.sh
# Before executing evalution code in order to calculate the metrics,
# please duplicate the content and style images to match the number of stylized images first. 
# (40 styles, 20 contents -> 800 style images, 800 content images)
python evaluation/copy_inputs.py --cnt PATH_FOR_CONTENT_IMAGES --sty PATH_FOR_STYLE_IMAGES
sh evaluation/eval.sh
```

### Testing
```
sh scripts/test.sh
```

### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  
```
sh scripts/train.sh
```

## Code explanation
The full model (fig. 2(a)) can be found at [MambaST.py](https://github.com/FilippoBotti/MambaST/blob/main/models/MambaST.py). In this file you can find the whole architecture. <br>
The Mamba Encoder/Decoder (fig. 2 (b) and fig. 2 (c)) module can be found at [mamba.py](https://github.com/FilippoBotti/MambaST/blob/main/models/mamba.py) <br>
Finally, our VSSM's implementation (both with a single input and with two input merged for style transfer) can be found at [mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/mamba_arch.py). If you want you can also find VSSM with different scans direction inside [single_direction_mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/single_direction_mamba_arch.py) and [double_direction_mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/double_direction_mamba_arch.py).

### Reference
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://www.arxiv.org/abs/2409.10385)<br> 


```
@InProceedings{Botti_2025_WACV,
    author    = {Botti, Filippo and Ergasti, Alex and Rossi, Leonardo and Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
    title     = {Mamba-ST: State Space Model for Efficient Style Transfer},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {7786-7795}
}
```

### Acknowledgments
Our code is inspired by [StyTR-2](https://github.com/diyiiyiii/StyTR-2) and [StyleID](https://github.com/jiwoogit/StyleID).
