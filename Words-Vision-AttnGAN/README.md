# Words-Vision-AttnGAN

<img src="framework.png"/>

### Dependencies

required packages: 

- `python 3.6`
- `Pytorch 1.0+`
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`

run `pip install -r requirements.txt` to install the required dependencies.

### Training

- Pre-train DAMSM models:
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/coco.yml --gpu 0`
 
- Train AttnGAN models:
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation our models.

### Sampling
- Run `python3 inference.py` to generate examples from caption. Results are saved to `outputs/`. 

### Pretrained Model
- [Text encoder for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `models/`
- [AttnGAN for coco](https://drive.google.com/open?id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi). Download and save it to `models/`

### Sample Outputs

<img src="sample_output.png"/>

