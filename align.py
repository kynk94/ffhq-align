import argparse
import glob
import os
from typing import List

import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ffhq_align import Aligner


class AlignDataset(Dataset):
    def __init__(self, images: List[str]) -> None:
        super().__init__()
        self.images = images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )

        self.n = len(self.images)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple:
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img) if self.transform is not None else img
        return path, img


def main() -> None:
    parser = argparse.ArgumentParser(description="Face Alignment")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="outputs")
    parser.add_argument("--resolution", "-r", type=int, default=512)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    args = vars(parser.parse_args())

    if os.path.isfile(args["input"]):
        images = [args["input"]]
    else:
        images = glob.glob(os.path.join(args["input"], "**", "*.*g"), recursive=True)
        images.sort()

    dataset = AlignDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=4,
    )
    aligner = Aligner(padding_mode="blur")
    if args["device"] in {"cuda", "gpu"}:
        aligner = aligner.cuda()

    pbar = tqdm.tqdm(dataloader)
    for paths, batch in pbar:
        aligned_images = aligner(batch, args["resolution"])

        for path, aligned in zip(paths, aligned_images):
            if aligned is None:
                continue

            if path == args["input"]:
                rel_path = os.path.basename(path)
            else:
                rel_path = os.path.relpath(path, args["input"])
            out_path = os.path.join(args["output"], rel_path)
            dirname = os.path.dirname(out_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            aligned = (
                aligned.mul(127.5)
                .add(127.5)
                .clamp(0, 255)
                .permute(1, 2, 0)
                .byte()
                .cpu()
                .numpy()
            )
            Image.fromarray(aligned).save(os.path.join(args["output"], rel_path))


if __name__ == "__main__":
    main()
