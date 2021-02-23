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

The API body must be of the a form data with the file key `file`

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
- [StoryGAN Model](https://www.google.com/). Download and save it to `models/storyGAN_model`

### Sample Outputs

<img src="sample_output.png"/>

