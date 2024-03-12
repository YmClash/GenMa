import os
from typing import List,Optional
from sentencepiece import SentencePieceProcessor

class Tokenizer:

    def __init__(self,model_path: Optional[str]):
        assert os.path.isfile(model_path),model_path
        self.sp_model = SentencePieceProcessor(model_path)

        self.n_words:int = self.sp_model.vocab_size()
        self.bos_id:int = self.sp_model.bos_id()
        self.eos_id:int = self.sp_model.eos_id()
        self.pad_id:int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self,s:str,bos:bool = True,eos:bool = False) -> List[int]:
        assert isinstance(s,str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self,t:List[int]) -> str :
        return self.sp_model.decode(t)