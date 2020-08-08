def get_vocab(vocab_path, tags_path):
    vocab = {}
    with open(vocab_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i  # to avoid the 0
        # loading tags (we require this to map tags to their indices)
    vocab['<PAD>'] = len(vocab) # 35180
    tag_map = {}
    with open(tags_path) as f:
        for i, t in enumerate(f.read().splitlines()):
            tag_map[t] = i 
    
    return vocab, tag_map

def get_params(vocab, tag_map, sentences_file, labels_file):
    sentences = []
    labels = []

    with open(sentences_file) as f:
        for sentence in f.read().splitlines():
            # replace each token by its index if it is in vocab
            # else use index of UNK_WORD
            s = [vocab[token] if token in vocab 
                 else vocab['UNK']
                 for token in sentence.split(' ')]
            sentences.append(s)

    with open(labels_file) as f:
        for sentence in f.read().splitlines():
            # replace each label by its index
            l = [tag_map[label] for label in sentence.split(' ')] # I added plus 1 here
            labels.append(l) 
    return sentences, labels, len(sentences)
