import torch
import h5py
import numpy as np
import math

from emg2qwerty.lightning import AutoEncoderModule
from emg2qwerty.transforms import Compose, ToTensor, Lambda

def pad(data):
    length = data.shape[-1]
    to_pad = length % 8
    if to_pad == 0:
        return data, length
    
    return (
        torch.nn.functional.pad(data, (0, 8 - to_pad)),
        length
    )

def gen_synth_data(
    input_path : str, # since i need transforms it should have a base session
    encoder : AutoEncoderModule,
    output_dir : str,
    noise: float = 0.1,
    device: str = "cuda"
):
    
    pipeline = Compose([
        ToTensor(fields=("emg_left", "emg_right"), stack_dim=-1),
        Lambda(lambda x : x.view(x.shape[0], -1)),
        Lambda(lambda x : x.permute(1, 0).unsqueeze(0))
    ])

    with h5py.File(input_path, 'r') as original, h5py.File(output_dir, 'w') as synthetic:

        # load emg tensor
        emg = pipeline(d).to(device)
        emg, length = pad(emg)

        with torch.no_grad():
            embedding = encoder.ac.enc(emg)
            n = torch.randn_like * noise
            synth = encoder.ac.dec(embedding + noise)
        
        synth = synth[..., :length]

        s = synth.squeeze(0).permute(1, 0).cpu().numpy()
        right = s[:, 16:]
        left = s[:, :16]

        synthetic.create_dataset("emg_right", data=right, compression='gzip')
        synthetic.create_dataset("emg_left", data=left, compression='gzip')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoEncoderModule.load_from_checkpoint("logs/auto_enc/checkpoints/epoch=47-step=46128.ckpt")
    model.eval()
    model.to(device)

    # Generate the data
    gen_synth_data(
        input_path="data/2021-07-22-1627004019-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
        output_dir="synth_data/1.h5py",
        encoder=model,
        noise=0.1,
        device=device
    )

if __name__ == "__main__":
    main()
