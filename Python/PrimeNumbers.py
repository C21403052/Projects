num1 = int(input("Please enter number 1: "))
num2 = int(input("Please enter number 2: "))


# check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


print("Prime numbers in the range", num1, "to", num2, "are:")
for num in range(num1, num2 + 1):
    if is_prime(num):
        print(num, end=" ")
print()
