import numpy as np


def build_dummy_text_features(max_text_len, word_dim=300, pos_dim=15):
    """Return fixed zero text features for pipelines that do not consume them."""
    sent_len = 2  # sos/eos
    seq_len = max_text_len + 2
    word_embeddings = np.zeros((seq_len, word_dim), dtype=np.float32)
    pos_one_hots = np.zeros((seq_len, pos_dim), dtype=np.float32)
    return word_embeddings, pos_one_hots, sent_len
