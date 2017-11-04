n = int(input())
s = map(int ,input().split())
stack = []
for num in s:
    for i in range(2, num):
        if (num % i) == 0:
            stack.append(num)
            break
    if i >= num-1:
        print(num,end=' ')
print()
print(*stack)