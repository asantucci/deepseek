import tiktoken

# reference: https://github.com/openai/tiktoken
class Tokenizer:
    def __init__(self, tokenizer_type: str):
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tiktoken.get_encoding(tokenizer_type)
        self.allowed_special = set()
    
    def get_token_id(self, token: str):
        '''
        get the token id
        '''
        return self.tokenizer.encode(token)
    
    
    def get_token_str(self, token_id: int):
        '''
        get the token string
        '''
        return self.tokenizer.decode([token_id])
        
    def add_special_tokens(self, tokens: list[str]):
        '''
        add the special tokens to the tokens
        '''
        vocab_size = self.tokenizer.n_vocab
        
        special_tokens = self.tokenizer._special_tokens
        for i, token in enumerate(tokens):
            if token not in special_tokens:
                special_tokens[token] = vocab_size + i
        
        enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name=f"{self.tokenizer_type}_im",
            pat_str=self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens=special_tokens
        )
        self.tokenizer = enc
        self.allowed_special = set(special_tokens.keys())
        
    def print_special_tokens(self):
        '''
        print the special tokens
        '''
        print(self.tokenizer._special_tokens)
    
    def get_vocab_size(self):
        '''
        get the vocab size
        '''
        return self.tokenizer.n_vocab
    
    def encode(self, text: str):
        '''
        encode the text
        '''
        return self.tokenizer.encode(text, allowed_special=self.allowed_special)
    
    def decode(self, token_ids: list[int]):
        '''
        decode the token ids
        '''
        return self.tokenizer.decode(token_ids)
    
if __name__ == "__main__":
    tokenizer = Tokenizer(tokenizer_type="cl100k_base")
    text = "Hello, how are you?"
    token_ids = tokenizer.encode(text)
    print(f"token_ids: {token_ids}")
    
    token_strs = tokenizer.decode(token_ids)
    print(f"token_strs: {token_strs}")
    
    tokenizer.add_special_tokens(["<|im_start|>"])
    text = "Hello, how are you? <|im_start|>"
    token_ids = tokenizer.encode(text)
    print(f"token_ids: {token_ids}")
    
    token_strs = tokenizer.decode(token_ids)
    print(f"token_strs: {token_strs}")
    
    