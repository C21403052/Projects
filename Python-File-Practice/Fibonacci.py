numterms = int(input("Enter number of terms you want to show: "))

a, b = 0, 1

for _ in range(numterms):
    print(a, end=" ")
    a, b = b, a + b

print()
