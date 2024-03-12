


with open('input.txt', 'r',encoding='utf-8') as file :
    data = file.read()

print(f'longeur du dataset: {len(data)}')
# print(data[:1000])

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f'nombre de Vocal : {vocab_size}')
print(''.join(chars))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s ]
decode = lambda l: ''.join([itos[i] for i in l])

enc = encode("Hallo comment cava ?")
print(enc)
print()
print(decode(enc))
