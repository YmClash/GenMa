import torch

with open('input.txt', 'r', encoding='utf-8') as file:
    dataset = file.read()

print(f'longeur du dataset: {len(dataset)}')
# print(data[:1000])

chars = sorted(list(set(dataset)))
vocab_size = len(chars)
print(f'nombre de Vocal : {vocab_size}')
print(''.join(chars))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(dataset), dtype=torch.long)
print(data[:10])
print(data.shape)
print(data.dtype)

"""
 preparation et split des donnée d'entrainement pour constituer 
 un set de d'entrainement  et set de validation 

"""
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8

print(len(data))
print(n)
print(train_data)
print(val_data)
train_data[:block_size + 1]

print(train_data)

enc = encode("Hallo comment cava ?")
msg = encode("Je me nomme toure mohamed ali")
print(f'msg length : {len(msg)} , :{msg}')
print()
print(f'message decode : {decode(msg)}\nmsg length {len(decode(msg))}')
