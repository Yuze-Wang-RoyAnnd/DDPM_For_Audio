# Pian Music Generation Using Diffusion Extention

# Diffwave
This project uses Diffwave as base model for extention, it is trained with 733,000 on a sample audio size of 14 different monophonic piano tacks. These piano track is sampled with 22.0 KHz and the Diffwave architecture takes about 5 second of sampled audio as an input.
We trained the model long enough to recover basic melody that resemble to piano sound, but not enough to reproduce meaningful note structure.

# Extention
In order to generate long audio samples, we decides to produce audio data in a two step process.
1. generate an base audio frame that is 5 seconds.
2. extend using the base audio frame.
Since extention problem is the same as an image inpainting problem, this is perfectly solvable with diffusion.

# Using
You can generate music by calling the sampling.py file as an CLI
```
python sampling.py path_to_model_dir
```
it support additional argument:
```
--ground_file: path to audio file that we want to generated from, if not provided, generate from static
--output: output file name, deafult=output.wav
--extent: inpaint algorithm you want to use, default=interpolation_extention
--seconds: how long do you want to extends
```

Or you can access it as an API
```
api_ = {"model_dir": "path_to_model_dir"}
model, device = sampling.setup(api_)
new_wav_2 = sampling.infer(model, sampling.normal_ext, "path_to_ground_file", device, 5)
```


# Interpolation gudiance
In the RePaint paper [Andreas et al.], the author introduced backtracking steps to harmonize the extended input and base input. We found this is wildly in efficient as it introduces additional diffusion steps.
We implemented interpolation gudiance to help leviate this problem. Instead of backtracking during diffusion steps, we harmonize the input of generated parts with base parts by adding them together using an lambda scale. We slowly change this scale during diffusion step as a way to guide the final output to acheve an harmonizing output.

Diffusion history of Interpolation gudiance:
![Diffusion history](DDPM_for_Audio/misc/interpolation.gif)

Diffusion history of Repaint:
![Diffusion history](DDPM_for_Audio/misc/repaint_concat.gif)

Diffusion history of Do-Nothing:
![Diffusion history](DDPM_for_Audio/misc/normal_concat.gif)

