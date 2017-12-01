n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
m = int(input())
cost = []
for _ in range(m):
    cost.append(list(map(int, input().split())))


def mul(a, b):
    somme = 0
    for i in range(len(a)):
        somme += a[i] * b[i]
    return somme


s = mul(a, b)


def swap(a, i, j):
    new_a = a.copy()
    x = a[i]
    new_a[i] = a[j]
    new_a[j] = x
    return new_a


count = 0
max = 0
k = 0
trans = []
for _ in range(n):
    for i in range(n):
        for j in range(i + 1, n):
            new_a = swap(a, i, j)
            count += cost[i][j]
            if mul(new_a, b) - s - count > max:
                k += 1
                max = mul(new_a, b) - s - count
                a = new_a
                trans.append([i, j])
            else:
                count -= cost[i][j]
print(k)


def pri(trans):
    for i in range(len(trans)):
        print(trans[i][0]+1, trans[i][1]+1)


pri(trans)
