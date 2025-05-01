student_scores = {
    'Aaron': 80,
    'Peter': 50,
    'John': 47,
    'Paul': 96
}

option = 0


def addstudent(name, grade):
    student_scores[name] = grade


def print_avg():
    total = 0
    for x in student_scores:
        grade = student_scores[x]
        total += grade
    average = total / len(student_scores)
    print("Score Average is: ", average)


def find_student():
    name = input("Enter Students name: ")
    if name in student_scores:
        print("Name: ", name)
        print("Grade: ", student_scores[name])
    else:
        print("Student not found please try again")


while option != 6:
    print("Please Pick An Option: ")
    print("1. Add Student")
    print("2. Show Average Score")
    print("3. Highest Score")
    print("4. Lowest Score")
    print("5. Search Student")
    print("6. Exit")
    option = int(input())
    if option == 1:
        sname = input("Enter Students name: ")
        sgrade = int(input("Enter Students Grade: "))
        addstudent(sname, sgrade)
    elif option == 2:
        print_avg()
    elif option == 3:
        print(max(student_scores.values()))
    elif option == 4:
        print(min(student_scores.values()))
    elif option == 5:
        find_student()
    elif option == 6:
        print("GoodBye")
    else:
        print("Error please select an option between 1-6")
        