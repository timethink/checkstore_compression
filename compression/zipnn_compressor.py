"""ZipNN compression (calls external script)."""

import os
import re
import subprocess
from typing import Optional

from .base import Compressor, CompressionResult
from .registry import register_compressor


@register_compressor("zipnn")
class ZipNNCompressor(Compressor):
    name = "ZipNN"

    def __init__(self, script_path: str = "", **kwargs):
        self.script_path = script_path

    def compress(self, filepath: str) -> Optional[CompressionResult]:
        if not self.script_path or not os.path.exists(self.script_path):
            print(f"Warning: ZipNN script not found at '{self.script_path}'")
            return None

        # Predict the auto-generated output filename
        if filepath.endswith(".safetensors"):
            auto_generated = filepath.replace(".safetensors", ".znn.safetensors")
        else:
            auto_generated = filepath + ".znn"

        if os.path.exists(auto_generated):
            os.remove(auto_generated)

        cmd = ["python", self.script_path, filepath, "--force"]
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )

            # Clean up generated file
            out_match = re.search(r"Compressed .*? to (.*?) using", result.stdout)
            if out_match:
                generated = out_match.group(1).strip()
                if os.path.exists(generated):
                    os.remove(generated)

            if os.path.exists(auto_generated):
                os.remove(auto_generated)

            # Parse ratio from stdout
            ratio_match = re.search(r"ratio is ([\d.]+)", result.stdout)
            if ratio_match:
                ratio = float(ratio_match.group(1))
                original_size = os.path.getsize(filepath)
                compressed_size = int(original_size * ratio)
                return CompressionResult(
                    original_size=original_size, compressed_size=compressed_size
                )
        except subprocess.CalledProcessError as e:
            print(f"ZipNN execution failed: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error running ZipNN: {e}")
        return None
