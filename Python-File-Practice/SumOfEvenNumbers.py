num = int(input("Please enter a positive number: "))
sumeven = 0
for x in range(1, num + 1):
    if x % 2 == 0:
        sumeven += x
print("Sum of even numbers: ", sumeven)
