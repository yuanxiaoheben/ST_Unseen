import pickle

def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode='wb') as handle:
        pickle.dump(data, handle)
def pad_data(seq_list, pad_val, max_len = None):
    
    if max_len == None:
        max_len = 0
        for seq in seq_list:
            if len(seq) > max_len:
                max_len = len(seq)
    new_seq_list = []
    for seq in seq_list:
        new_seq = seq
        if len(seq) < max_len:
            new_seq = new_seq + [pad_val for x in range(max_len - len(seq))]
        new_seq_list.append(new_seq)
    return new_seq_list