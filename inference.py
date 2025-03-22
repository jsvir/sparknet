from model import EncDecClassificationModel
import torch
import librosa
from glob import glob
import argparse


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--wavs_dir', type=str, default="wavs")
    parser.add_argument('--model_path', type=str, default="ckpt/kws_C_16.ckpt")

    args = parser.parse_args()

    model = EncDecClassificationModel.load_from_checkpoint(args.model_path)
    model.eval()

    for wav in glob(f"{args.wavs_dir}/*.wav"):
        waveform = torch.tensor(librosa.load(wav, sr=model.cfg.sample_rate)[0]).float().reshape(1, -1)
        with torch.no_grad():
            outputs = model(
                input_signal=waveform,
                input_signal_length=torch.ones(waveform.size(0)) * waveform.size(1))
        logits = outputs[0].squeeze()
        prediction = logits.argmax(-1).item()
        print(f"Predicted: {model.cfg.labels[prediction]} for wave {wav}")


if __name__ == '__main__':
    main()
