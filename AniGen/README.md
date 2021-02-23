# AniGen

### Dependencies

required packages: 

- `python 3.6`
- `pytorch 1.6`
- `tensorflow 1.15`


run `pip install -r requirements.txt` to install the required dependencies.

### APIs

- Three API endpoints are created.
1. 'gpt_train' - To train the gpt2 model with the story input. 
The input file must be of the format:
```
text content <|endoftext|>
text content <|endoftext|>
```
The text data must be followed by an `<|endoftext|>` tag.

The API takes in form data with the file key `file`. It accepts `.txt` files.

2. 'gpt_sample' - To produce sample from the traned gpt2 model.
The API body must be of the json format:
```
{
    "input":"input text"
}
```

3. 'storygan_sample' - To produce animation from an imput text. 
The API body must be of the Json format:
```
{
    "input":"Fred is driving"
}
```

### Pretrained Model
- [StoryGAN Model](https://drive.google.com/file/d/1UYwAoVDP37Vz0k3lJtTH_tTrUhiy0Ch2/view?usp=sharing). Download and save it to `models/storyGAN_model`

### Sample Output
- input: `Fred is driving`

<img src="output/out.gif"/>