import ijson
import numpy as np
import os


def convert_json_to_bin(json_path, bin_path, dtype=np.uint32, chunk_size=1_000_000):
    """
    Convertit un fichier JSON {"tokens": [...]} en fichier binaire brut,
    sans jamais charger la liste complète en RAM.
    """
    itemsize = np.dtype(dtype).itemsize
    n_written = 0

    with open(json_path, 'rb') as f_in, open(bin_path, 'wb') as f_out:
        parser = ijson.items(f_in, 'tokens.item')  # stream élément par élément

        buffer = []
        for token in parser:
            buffer.append(token)
            if len(buffer) >= chunk_size:
                arr = np.array(buffer, dtype=dtype)
                f_out.write(arr.tobytes())
                n_written += len(buffer)
                buffer = []

        # dernier chunk incomplet
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            f_out.write(arr.tobytes())
            n_written += len(buffer)

    print(f"{n_written} tokens écrits dans {bin_path} ({n_written * itemsize / 1e9:.2f} GB)")

convert_json_to_bin(os.path.join("datas\\tokenized", "tokens.json"), os.path.join("datas\\tokenized", "tokens.bin"), dtype=np.uint32)