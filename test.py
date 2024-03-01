import torch
import math

def positional_encoding(seq_len, d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe[:seq_len, :]

# 예시 파라미터
seq_len = 10 # 입력 시퀀스 길이
d_model = 512 # 모델 차원

# 포지셔널 인코딩 생성
pe = positional_encoding(seq_len, d_model)

print(pe)
print(pe.shape) # 출력: [seq_len, d_model]

# 임베딩 벡터 생성을 위한 임의의 입력 인덱스
input_indices = torch.randint(0, 1000, (seq_len,))

print(input_indices)

# 임베딩 레이어 생성
embedding = torch.nn.Embedding(1000, d_model)

# 임베딩을 통해 입력 벡터 생성
input_vectors = embedding(input_indices)

print(input_vectors)

# 포지셔널 인코딩을 입력 벡터에 더함
encoded_vectors = input_vectors + pe

print(encoded_vectors.shape) # 출력: [seq_len, d_model]

print(-0 == 0.0)


import torch

zero = torch.tensor(0.0)
neg_zero = torch.tensor(-0.0)

# 값 비교
print(zero == neg_zero)  # True를 출력

# 연산 확인
print(zero + neg_zero)  # 0.0을 출력
print(zero - neg_zero)  # 0.0을 출력