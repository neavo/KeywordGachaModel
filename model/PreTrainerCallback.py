import os
import json

from transformers import TrainerCallback

class PreTrainerCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_save(self, args, state, control, **kwargs):
        path = args.output_dir

        for f in os.scandir(args.output_dir):
            if f.is_dir() and f.name.startswith("checkpoint"):
                os.makedirs(f.path, exist_ok = True)
                with open(f"{f.path}/training_args.json", "w", encoding = "utf-8") as file:
                    json.dump(args.to_dict(), file, indent = 4, ensure_ascii = True)