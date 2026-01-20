from models.NextMemQwen import NextMemQwen
from config import DEFAULT_CONFIG

text = """
Alice is currently a lawyer at a top-tier law firm in Manhattan.
Her professional focus is Intellectual Property and technology law.
She lives in the Upper West Side. She likes artisanal lobster rolls.
She went to the Metropolitan Opera yesterday.
She will be attending a high-profile court hearing tomorrow.
"""
def example():
    nextmem = NextMemQwen(DEFAULT_CONFIG['NextMemQwen'])
    latent_memory = nextmem.encode(text)
    decoded_text = nextmem.decode(latent_memory)
    print(decoded_text)

if __name__ == '__main__':
    example()
