import regex as re
import get_data_set as gdt
import json, os

class Tokenizer():

    def __init__(self, vocab_size=8192):
        self.patterns = re.compile(
            r"(?i)"
            r"<\|(?:who_i_am|end_who_i_am|bos|eos|pad)\|>|"                       # tokens spéciaux
            r".?(?:[cdjlmnst]|qu)'|'(?:[sdmt]|ll|ve|re)|"                     # contractions
            r"[^\r\n\p{L}\p{N}]?+\p{L}++|"                                    # mots (lettres)
            r"\p{N}{1,3}+|"                                                   # nombres
            r" ?[^\s\p{L}\p{N}<>]++[\r\n]*+|"                                 # ponctuation **sans < et >**
            r"\s++$|\s*[\r\n]|\s+(?!\S)|\s"
        )
        base_bytes = 256
        special_tokens = ["<|who_i_am|>", "<|end_who_i_am|>", "<|bos|>", "<|eos|>", "<|pad|>"]
        self.special_tokens_map = {token: base_bytes + i for i, token in enumerate(special_tokens)}

        self.reserved_after_special_tokens = base_bytes + len(special_tokens)

        self.vocab_size = vocab_size
        self.num_merges = vocab_size - self.reserved_after_special_tokens

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        self.isVocabLoaded = False
        self.isMergesLoaded = False

        if(os.path.exists("merges.json")):
            self.merges = load_merges("merges.json")
            self.isMergesLoaded = True

        if os.path.exists("vocab.json"):
            with open ("vocab.json", "r", encoding="utf-8") as f:
                self.vocab = load_vocab("vocab.json")
                self.isVocabLoaded = True

    def split_text(self, text):
        return re.findall(self.patterns, text)
    
    def convert_segments_to_utf8(self):
        """
        segments: list[str] (tokens produits par split_text)
        specials_map: dict[token_str] -> reserved_id (int)
        retourne: list[list[int]] where each inner list is either:
        - [reserved_id] for a special token
        - list(byte values) for normal tokens
        """
        out = []
        for segment in self.segments:
            if segment in self.special_tokens_map:
                out.append([self.special_tokens_map[segment]])
            else:
                # ordinary token: bytes -> list of ints
                out.append(list(segment.encode("utf-8")))
        return out
    
    def get_stats(self):
        counts = {}
        for segment in self.segments:

            for pair in zip(segment, segment[1:]): # python way to iterate 2 consecutive elts
                counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, pair, idx):
        newsegments = []
        for segment in self.segments:
            newsegment = []

            i = 0
            while i < len(segment):
                if i < len(segment) -1 and segment[i] == pair[0] and segment[i+1] == pair[1]:
                    newsegment.append(idx)
                    i += 2
                else:
                    newsegment.append(segment[i])
                    i += 1
            newsegments.append(newsegment)
        self.segments = newsegments

    def tokenize(self, text):

        self.segments = self.split_text(text)
        self.segments = self.convert_segments_to_utf8()

        self.isVocabLoaded = True
        self.isMergesLoaded = True
        for i in range(self.num_merges):
            
            stats = self.get_stats()
            pair = max(stats, key=stats.get)
            idx = self.reserved_after_special_tokens + i

            for tok ,id_ in self.special_tokens_map.items():
                self.vocab[id_] = tok.encode("utf-8")
            for (p0, p1), idx2 in self.merges.items():
                self.vocab[idx2] = self.vocab[p0] + self.vocab[p1]
            print("merging {} -> {} ({})".format(pair, idx, self.decode(pair)))

            self.merge(pair, idx)
            self.merges[pair] = idx
        

        save_merges(self.merges, "merges.json")
        save_vocab(self.vocab, "vocab.json")


    def decode(self, ids):
            if(not self.isVocabLoaded or not self.isMergesLoaded):
                print("You should use #tokenize first to generate vocab and merges")
                return
            
            tokens = b"".join(self.vocab[idx] for idx in ids)
            text = tokens.decode("utf-8", errors="replace")
            return text
    
    def get_stats_for_encode(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): # python way to iterate 2 consecutive elts
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge_for_encode(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):

        if(not self.isVocabLoaded or not self.isMergesLoaded):
            print("You should use #tokenize first to generate vocab and merges")
            return

        # Étape 1: découpage identique à l'entraînement
        segments = re.findall(self.patterns, text)

        # Étape 2: conversion en entiers (gérer specials)
        tokens = []
        for seg in segments:
            if seg in self.special_tokens_map:
                tokens.append(self.special_tokens_map[seg])
            else:
                tokens.extend(list(seg.encode("utf-8")))

        while(len(tokens) >= 2):
            stats = self.get_stats_for_encode(tokens) #only care 'bout the keys
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) #for any pair inside stats looking into merges in what index it has, get the pair with the lowest index
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge_for_encode(tokens, pair, idx)
        return tokens


def save_vocab(vocab, path):
    vocab_serializable = {str(k): v.decode('latin-1') for k, v in vocab.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

def save_merges(merges, path):
    merges_serializable = [(a, b, new_id) for (a, b), new_id in merges.items()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab_loaded = {int(k): v.encode('latin-1') for k, v in json.load(f).items()}
        return vocab_loaded

def load_merges(path):
    with open(path, "r", encoding="utf-8") as f:
        merges_list = json.load(f)
        merges_loaded = {(a, b): new_id for a, b, new_id in merges_list}
        return merges_loaded

if __name__ == '__main__':

    tkn = Tokenizer(24576)
    tkn.tokenize(gdt.gather_datas())
    mesg = tkn.encode("Mais quel gros tas ! T'es même pas capable de farmer tes primogems toi même gros singe ?! -_-")
    print(mesg)
    print(tkn.decode(mesg))
