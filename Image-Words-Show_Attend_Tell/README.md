# Image-Words-Show_Attend_Tell

<img src="model.png"/>

### Dependencies

required packages: 

- `python 3.6`
- `Pytorch 1.5`
- `scikit-image`
- `matplotlib`
- `scipy`

run `pip install -r requirements.txt` to install the required dependencies.


### Inference
- Run `python3 caption.py --img='PATH_TO_IMAGE' --model='./PATH_TO_MODEL/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='./PATH_TO_WORDMAP/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5` to generate examples caption. The results will be opened in a window.

### Training

Inputs files required for training are:
- HDF5 file containing images 
- JSON file for each encoded caption
- JSON file for the caption lengths
- JSON file which contains the word map

To train your model from scratch, simply run this file:
- `python train.py`

### Pretrained Model
- You can download this pretrained model and the corresponding word_map [here](https://drive.google.com/open?id=189VY65I_n4RTpQnmLGj7IzVnOF6dmePC)
- Put the downloaded model in the 

### Sample Outputs

<img src="sample_output.png"/>
