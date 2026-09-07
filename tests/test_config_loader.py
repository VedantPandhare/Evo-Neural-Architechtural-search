import os
import tempfile

from configs.loader import load_config


def test_load_config_reads_yaml_dict():
    content = "population_size: 5\nepochs: 2\ndevice: cpu\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg == {"population_size": 5, "epochs": 2, "device": "cpu"}
    finally:
        os.unlink(path)
