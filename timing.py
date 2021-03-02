def prime(n):
    for i in range(2, int(n ** (1 / 2))):
        if n % i == 0:
            return False
    return True

n = int(input())
for i in range(2, int(n ** (1/2))):
    if n % i == 0:
        if prime(i)
