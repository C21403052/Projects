from Emotion_Identifier import emotion_identifier
if __name__ == '__main__':
    # While loop to take in user input
    while True:
        print("Select an option: ")
        print("1. Record a Session")
        print("2. Quit\n")
        option = int(input())
        if option == 1:
            # Run to test model performance1
            emotion_identifier()
        elif option == 2:
            print("Program Shutting Down!!!")
            break