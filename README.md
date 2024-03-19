# Reactive StyleGAN NDI

![Reactive StyleGAN NDI](_md-assets/banner.png)

## What is this? 

This is a collection of thingamajigs to create real-time stylegan3 inferences based on input values in Touchdesigner.

Touchdesigner sends OSC data to a python program which in turn sends its created images back via NDI.

On a laptop 4090 (Windows) I get around 23 frames/s on 1024x1024 models and around 40 frames/s for 512x512 models.

## osc port in: 161

| osc path | description | type |
| ------------- | ------------- | ------------- |
| /setpkl | set pkl path to be loaded | string |
| /latentX | pushes the latent in x direction | float |
| /latentY | pushes the latent in y direction | float |
| /truncpsi | psi truncate | float |
| /trunccutoff | psi trunc cutoff | int |
| /inputtransformx | input transform x-axis | float |
| /inputtransformy| input transform y-axis | float |
| /inputtransformrotation| input transform rotation | float |
| /imagescaledb | image scale db | float |
| /get_favorite_data | gets momentary latent information | float |
| /set_latent_offset | sets latent offsetto match a favorite | float |
| /fps | sets desired render frequency | int |
| /randomize | randomizes the seed | any |


## osc port out: 162

Receive favorite data

| osc path | description | type |
| ------------- | ------------- | ------------- |
| /favorite_data | favorite data of momentary latents | string |

example favorite

```json
{"latentX":161.161, "latentY":13.12}
```

## osc port out: 163

Receive model loading info

| osc path | description | type |
| ------------- | ------------- | ------------- |
| /model_loading | 1 if loading, -1 if done loading pkl model | int |

## Installation

Start off with getting an environment ready to run stylegan3.

A great test to see if everything is peachy is to run 

```shell
python visualizer.py
```

If you can generate images with it, you're fine.

Then install the additional requiremens listed in [additional-requirements.txt](additional-requirements.txt)
(and let me know if something is missing)

I also included my working conda environment, which potentially includes unnneccessary stuff here: [environment.yml](environment.yml)

## Running it

Go ahead and let it run with

```shell
python lazer_run.py
```

# In Touchdesigner

Open up [TD\gan-minclient.toe](TD\gan-minclient.toe) - it should start generating as soon as you have chosen your model folder in the model folder parameter of the big component. This component is also available as a .tox [TD\reactive-stylegan-ndi-v0.1.tox](TD\reactive-stylegan-ndi-v0.1.tox)

It's set up to react to joystick, but you can switch to some basic audio reactivity with the big joystick/music button on the left. Enjoy!