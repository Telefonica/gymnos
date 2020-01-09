#
#
#   Ask utils
#
#


def confirm(text, default=True):
    possible_options = "Y/n" if default else "y/N"

    while True:
        user_input = input(text + " ({}): ".format(possible_options))

        if user_input in ["yes", "y"]:
            return True

        if user_input in ["no", "n"]:
            return False

        print("Invalid option {}".format(user_input))


def text(text, required=False, default=None):
    additional_text = "({}): ".format(default) if default is not None else ": "
    while True:
        user_input = input(text + additional_text)

        if user_input:
            return user_input

        if not required:
            return default

        print("Required option")
