from responses import responses_dict, options_dict, line_one, line_two, line_one_options

exit_commands = ("quit", "goodbye", "exit", "no", "bye")
help_options = ("help", "options")

class ChatBot:
  
    #define .make_exit() below:
    def make_exit(self, user_message):
        for command in exit_commands:
            if command in user_message:
                print("Have a wonderful day!")
                exit()

    def bot_greeting(self):
        return print(line_one + line_two)

    # Define Options for User So they can call on them again whenever 
    def user_options(self, user_message):
        if user_message in help_options:
            return print(f"{line_one_options}{options_dict}\n")
            
    #define .chat() below:
    def chat(self):
        self.bot_greeting()
        self.user_options("help")
        user_message = input("Your input: ")
        self.chat_handler(user_message)
    
    # define a function to manage the loop and be central to the conversation
    def chat_handler(self, user_message):
        while not self.make_exit(user_message):
            if user_message in help_options:
                self.user_options("help")
            self.choice_selection(user_message)
            user_message = input("\nYour input: ")
            self.chat_handler(user_message)

    # function to handle the selection of choices from user
    def choice_selection(self, user_message):
        if user_message in responses_dict:
            print(responses_dict.get(user_message))

 

#initialize ChatBot instance below:
bot = ChatBot()

#call .chat() method below:
bot.chat()

