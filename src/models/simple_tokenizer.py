class SimpleTokenizer:
    """Simple tokenizer for humor detection."""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_word = {0: '<pad>', 1: '<unk>'}
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        word_freq = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size - 2 (accounting for <pad> and <unk>)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        idx = 2  # Start after <pad> and <unk>
        for word, freq in sorted_words[:self.vocab_size - 2]:
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            idx += 1
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
    
    def encode(self, text, max_length=128):
        """Encode text to token indices."""
        words = text.lower().split()
        indices = []
        
        for word in words[:max_length]:
            indices.append(self.word_to_idx.get(word, 1))  # 1 is <unk>
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(0)  # 0 is <pad>
        
        return indices[:max_length]
