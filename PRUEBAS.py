
with open('pruebas.txt', 'r') as f:
    lines = f.readlines()

A = [l.split(',') if l is not None else l for l in lines]

print(A)
print(A[0][1].strip())