string = input("Enter a string: ")
vowel_count = 0
for char in string:
    if char.lower() in 'aeiou':
        vowel_count += 1
print("Number of Vowels in the string: ", vowel_count)
