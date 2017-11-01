#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Artificial Intelligence
Prof. Björn Ommer
WS17/18

Exercise sheet 1 - Question 1

@author: Uta Büchler
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

random.seed()

class Chatbot():
    
    def __init__(self,greetings,questions,reaction,farewells,questions_user=None,answers=None):
        '''
        Initialize the chatbot given the input arguments
        (is called when creating an object of the class Chatbot)
        
        input:  greetings:      list of strings - predefines which greetings the bot can use
                questions:      list of strings - predefines the questions the bot can ask
                reaction:       function - decides how to react to the answers made by the user
                questions_user: list of strings - predefines which questions the user might ask the bot (Ex. 1b)
                answers:        list of strings - predefines the answers of the bot given a specific question of the user
                farewells:      list of strings - predefines how the bot can say goodbye
        
        '''
        self.reaction = reaction#function 
        self.greetings = greetings
        self.questions = questions
        self.questions_user = questions_user
        self.answers = answers
        self.farewells = farewells
        #during the conversation the bot has to ask all questions
        #question index is removed in askQuestion() if question was asked
        self.notAsked = range(0,len(questions))
        self.shownGauss = 0#boolean - has to be changed to 0 for Question 1c
    
    def greeting(self):
        '''
        Question 1a)
        The bot outputs randomly a greeting given all possible greetings in self.greetings
        Afterwards, the bot asks for the name of the user and reacts by saying 'Nice to meet you USER',
        where USER has to be replaced by the input made from the user.
        
        Question 1b)
        After greeting the user the Bot has to check if the user asked questions during his last input.
        If yes, he has to answer them. For extracting the different parts of the users input (in case
        he asked a question) you can use self.extractParts(user_input,delimiter)
        '''
        # choose a random greeting
        index = np.random.randint(len(self.greetings))
        greeting = self.greetings[index]
        print(greeting)
        
    def askQuestion(self):
        '''
        Question 1a)
        The bot outputs randomly a question given all possible questions in self.questions
        If the question was asked, it should be removed from the variable self.notAsked.
        Afterwards, the bot should react to the answer of the user by using the input of the user.
        Example: 
            Question_bot: What are your hobbies? 
            Answer_user: football, meeting friends
            Reaction_Bot: I don't like football, but I like meeting friends
            
        How the bot reacts to which question has to be given by the function self.reaction
        For extracting the different parts of the answer (1st part is 'football', second is 'meeting friends')
        you can use the function self.extractParts()

        Question 1b)
        After reacting to the answer of the user the Bot has to check if the user asked questions during his last input.
        If yes, he has to answer them. For extracting the different parts of the users input (in case
        he asked a question) you can use self.extractParts(user_input,delimiter)
        '''
        # choose a random question and delete it from not asked questions
        index = np.random.choice(self.notAsked)
        question = self.questions[index]
        self.notAsked = np.delete(self.notAsked, np.where(self.notAsked == index))

        # users answer
        answer_user = input(question + '\n')

        # bots reaction
        self.reaction(self, question, answer_user)

        parts = self.extractParts(answer_user, ' ')

        if self.isQuestion(parts):
            answers = self.answerQuestion(parts)
            for answer in answers:
                print(answer)



    def farewell(self):
        '''
        Question 1a)
        The bot outputs randomly a farewell given all possible greetings in self.farewells
        '''
        # choose a random farewell
        index = np.random.randint(len(self.farewells))
        farewell = self.farewells[index]
        print(farewell)
                                 
    def extractParts(self,user_input,delimiter,outputType = str):
        '''
        Find the different parts of the users input, by splitting the string 'user_input' in its parts
        using the delimiter specified as input. If necessary, the output can be of a different type than
        string (for example float)
        
        Input:
            user_input: input made by the user
            delimiter:  substring which identifies the end and start of a part
            outputType: type of the elements in the output list 'parts'
        
        Output:
            parts: list of strings - contains the parts of the user's input in the sepecified type
        '''
        parts = [outputType(x.strip()) for x in user_input.split(delimiter)]
        
        return parts
    
    def answerQuestion(self,user_input):
        '''
        Question 1b)
        This function finds the appropriate answer to the question(s) made by the user.
        First, the question has to be find in self.questions_user. This then leads to the index
        needed to find the right answer of the bot using self.answers.
        
        Input: user_input: list of strings - contains the parts of the user's input
        Output: answers: all answers to the questions made from the user in his/her current input
    
        '''
        answers = []

        # find appropriate answers
        for idx, question in enumerate(self.questions_user):
            question_parts = self.extractParts(question, ' ')
            if any(string in user_input for string in question_parts):
                answers.append(self.answers[idx])

        return answers

    def isQuestion(self,user_input):
        '''
        Question 1b)
        This function checks if the user asked a question during his last input.
        Input:  user_input: list of strings - contains the parts of the user's input
        Output: a boolean variable, stating if one of the user_input parts contains a questionmark or not
        '''
        # find question marks
        if any('?' in string for string in user_input):
                return True

    def plotGauss(self):
        '''
        Question 1c)
        This function let the chatbot plot two Gauss Curves given a mean and two different standard deviations
        chosen by the user.
        
        Use the function "norm" from the library scipy.stats to compute the function values y1 and y2
        given x, the mean and the two standard deviations std1 and std2
        Plot both functions (as subplots) using the library matplotlib.pyplot.
        Play a little bit around with the properties one can set for the plots
        (like setting a title,changing the line color etc etc)
        '''
        self.shownGauss=1
        user_input=input("Do you want to see how a Gauss curve looks like? \n")
        if user_input.lower()=='yes':
            user_input = input("Cool! Then you need to give me a random number between 0 and 1 for the mean \n")
            mean = float(user_input)
            user_input = input("I want to show you 2 Gaussians with different standard deviations. So I need two numbers now between 0 and 1 \n")
            stds = self.extractParts(user_input,',',outputType=float)
            
            # set x range of plot
            x1 = np.linspace(mean - 5*stds[0], mean + 5*stds[0], 100)
            x2 = np.linspace(mean - 5*stds[1], mean + 5*stds[1], 100)

            # set y values
            y1 = norm.pdf(x=x1, loc=mean, scale=stds[0])
            y2 = norm.pdf(x=x2, loc=mean, scale=stds[1])

            # plot figure
            plt.figure()
            plt.subplot(121)
            plt.plot(x1, y1, 'b')

            plt.subplot(122)
            plt.plot(x2, y2, 'r')
            plt.show()

            
        elif user_input.lower()=='no':
            print("Oh... ok.")
            
    def main(self):
        '''
        main function to perform the conversation between chatbot and user
        starts with greeting, then asks all available questions and ends with a farewell
        
        The part with the Gauss curve only applies for Question 1c.
        Change the boolean variable self.shownGauss to 0, so that the function plotGauss will be executed
        during the conversation.
        '''
        self.greeting()
        while not len(self.notAsked)==0:#ask questions until all questions are asked
            self.askQuestion()
            if not self.shownGauss:#plot Gauss if it hasn't been done already
                prob = 1 if len(self.notAsked)==0 else random.randint(0,1)#plot with probability of 50% or 100% if last question was asked
                if prob: self.plotGauss()
        self.farewell()

if __name__ == "__main__":

    def reaction(bot,question,answer):
        '''
        Question 1a)
        Determines the reaction of the chatbot to an answer given by the user after the bot asked a question
        Input:
            bot:        Object of the class Chatbot
            question:   string - containing the question the bot asked
            answer:     string - containing the answer of the user to the question of the bot
        Output: nothing - the output/reaction should be printed in the console
        '''
        parts = bot.extractParts(answer, '?')

        notAnswered = True

        while notAnswered:
            if question == bot.questions[2]:
                print('Your dad is 44!')
                notAnswered = False
            elif any('4' in string for string in parts):
                print('Minus 1 thats 3 quick math.')
                notAnswered = False
            elif any('ye' in string for string in parts):
                print('Aight, Boom, Lets go!')
                notAnswered = False
            else:
                answer = input('Mans gonna come back in. \n')
                if 'Lets go!' in answer:
                    notAnswered = False

    bot = Chatbot(greetings = ['Yo, Gah!','Boom!', 'And da thing goes skraaaaaaaaaa!'],
              questions = ['2 + 2 is?','You feeling it?','How old is your dad?'],
              reaction = reaction,
              questions_user = ['Why dont you take off your jacket?','What?', 'Got ketchup?', 'Are you ready for this Gah?'],
              answers =  ['Mans not hot!','You dickheaaaaad!','Raw sauce! No ketchup!', 'Come on I was born ready.' ],
              farewells = ['You dont know.','Take my twix by force.','Hold tight!'])

    bot.main()
