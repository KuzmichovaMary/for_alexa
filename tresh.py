n, w = map(int, input().split())
left, right = map(int, input().split())
coords = sorted(list(map(int, input().split())))
c = []
i1 = 0
i2 = 0
m = 0
if left - right < w:
    print(-1)
elif left - right == w:
    print(len(coords))
    for i in coords:
        print(i)
else:
    