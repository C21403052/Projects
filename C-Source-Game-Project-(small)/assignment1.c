/*
Author: Aaron Burton
Date: 11/11/2021
Description: This program will simulate a math quiz game. It will have a 
menu that will display a list of options. The menu will have 1. Asking the 
user how many questions they would like to enter, Maximuim being 5 questions.
2. Will start the quiz and allow the user to answer the questions. 3. Will 
display the correct and incorrect answers, this option will only be available 
after the quiz is completed. Finally 4. will exit the program.
*/
#include <stdio.h>

int main()
{//start main

    //declare variables
    int option = 0;
    int questions = 0;
    int i = 0;
    int correctCounter = 0;
    int incorrectCounter = 0;
    int A = 0;
    int input = 0;
    int QInput = 0;
    
    //Display the menu
    printf("|*****************************|\n");
    printf("|WELCOME TO THE TUD MATHS QUIZ|\n");
    printf("|_____________________________|\n");

    //while loop to present menu until user exits program
    while(option != 4)
    {//start while loop

        //Display Menu
        printf("\nPlease select an option between 1-4:\n");
        printf("1. Choose Number of Question's (Max 5)\n");
        printf("2. Start Quiz\n");
        printf("3. Show Results\n");
        printf("4. Exit Game\n");

        //take in user input
        scanf("%d", &option);
        //buffer to prevent the input of a char from looping the same line of code
        while (getchar() != '\n');

        //check if the number entered was between 1-4
        if(option > 4)
        {//start if
            printf("\nERROR INVALID INPUT\n");
        }//end if
        else if(option < 1)
        {//start else if
            printf("\nERROR INVALID INPUT\n");
        }//end else if

        //switch statement to preform action chosen
        switch (option)
        {//start switch
        
        //case 1 will take the amount of questions the user would like in the quiz
        case 1:
            //while loop to loop the input until the user inputs a number between 1-5
            while(questions == 0)
            {//start while loop
                printf("How many questions would you like to have? (1-5)\n");
                //take in user input
                scanf("%d", &QInput);

                

                /*if statements to check if the input is in the range of 1-5 
                and if so assign the value to the variable questions*/
                if(QInput<6)
                {//start if
                    if(QInput>0)
                    {//start if
                        questions = QInput;
                    }//end if
                }//end if
                else
                {//start else
                    printf("\nInvalid Input please select a number between 1-5\n");
                }//end else
            }//end while loop
            printf("You have selected %d questions.\n", questions);
            break;

        //case 2 will start the quiz
        case 2:
            //if statement to prevent the user from starting the quiz if they have not select an amount of questions
            if(questions == 0)
            {//start if
                printf("\nPlease select an amount of questions you would like in the quiz\n");
                break;
            }//end if

            //reset counter at thhe start of each quiz
            correctCounter = 0;
            incorrectCounter = 0;

            //for loop to go through each question up to the amount the user selected
            for(i = 0; i < questions; i++)
            {//start for loop

                //switch statement to ask the user each question
                switch(i)
                {//start switch
                    //Question 1
                    case 0:
                        A=20/5;
                        printf("20/5=?\n");
                        scanf("%d", &input);

                        //buffer to prevent the input of a char from looping the same line of code
                        while (getchar() != '\n');

                        //if statement to check if the input is correct
                        if(input == A){//start if
                            printf("Correct\n");
                            correctCounter++;
                        }//end if

                        else{//start else
                            printf("Incorrect\n");
                            incorrectCounter++;
                        }//end else

                        break;

                    //Question 2
                    case 1:
                        input = 0;
                        A=(12/4)*3;
                        printf("(12/4)*3=?\n");
                        scanf("%d", &input);

                        //buffer to prevent the input of a char from looping the same line of code
                        while (getchar() != '\n');

                        //if statement to check if the input is correct
                        if(input == A){//start if
                            printf("Correct\n");
                            correctCounter++;
                        }//end if

                        else{//start else
                            printf("Incorrect\n");
                            incorrectCounter++;
                        }//end else

                        break;

                    //Question 3
                    case 2:
                        input = 0;
                        A=(77/7 +4)*3;
                        printf("(77/7 + 4)*3=?\n");
                        scanf("%d", &input);

                        //buffer to prevent the input of a char from looping the same line of code
                        while (getchar() != '\n');

                        //if statement to check if the input is correct
                        if(input == A){//start if
                            printf("Correct\n");
                            correctCounter++;
                        }//end if

                        else{//start else
                            printf("Incorrect\n");
                            incorrectCounter++;
                        }//end else
                        break;
                    
                    //Question 4
                    case 3:
                        input = 0;
                        A=5+5;
                        printf("5+5=?\n");
                        scanf("%d", &input);

                        //buffer to prevent the input of a char from looping the same line of code
                        while (getchar() != '\n');

                        //if statement to check if the input is correct
                        if(input == A){//start if
                            printf("Correct\n");
                            correctCounter++;
                        }//end if

                        else{//start else
                            printf("Incorrect\n");
                            incorrectCounter++;
                        }//end else
                        break;
                    
                    //Question 5
                    case 4:
                        input = 0;
                        A=10%3;
                        printf("What is the remainder of 10/3=?\n");
                        scanf("%d", &input);

                        //buffer to prevent the input of a char from looping the same line of code
                        while (getchar() != '\n');

                        //if statement to check if the input is correct
                        if(input == A){//start if
                            printf("Correct\n");
                            correctCounter++;
                        }//end if

                        else{//start else
                            printf("Incorrect\n");
                            incorrectCounter++;
                        }//end else
                    break;
                }//end switch
            }//end for loop
        
        //display the amount the user got correct and incorrect
        case 3:
            printf("These corrections are based on your last quiz.\n");
            printf("Answers Correct: %d\n", correctCounter);
            printf("Answers Incorrect: %d\n", incorrectCounter);
            break;

        default:
            printf("Invalid input please enter a number between 1-4");
            break;

        }//end switch
        
    }//end while loop

    //Goodbye Message
    printf("\nGOODBYE :)");
    return 0;
}//end main