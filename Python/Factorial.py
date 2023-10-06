num = int(input("Please enter a positive number: "))
temp = num
for x in range(1, num):
    num = num * (temp - x)
print(num)
