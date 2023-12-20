import os
import random
import datetime

from gradio_prompter.sdxl_prompt_constants import prompt_directory


def generate_unique_filename():
    current_time = datetime.datetime.now()
    random_number = random.randint(1000, 9999)
    filename = current_time.strftime("prompt_log_%Y%m%d_%H%M%S") + f"_{random_number}.txt"
    return filename


def get_random_element(element_list):
    """Returns a random element from a given list."""
    if not element_list:
        return None  # Return None if the list is empty
    return random.choice(element_list)


def remove_quotes(s):
    # Define the characters to be removed
    quotes = {'"', "'"}

    # Find the first non-quote character from the start
    start = 0
    while start < len(s) and s[start] in quotes:
        start += 1

    # Find the first non-quote character from the end
    end = len(s) - 1
    while end >= 0 and s[end] in quotes:
        end -= 1

    # Slice the string to remove the quotes
    return s[start:end + 1]


def select_random_line(filename: str):
    selected_line = None
    line_number = 0
    with open(os.path.join(prompt_directory, filename), 'r') as file:
        for i, line in enumerate(file, 1):
            if random.random() < 1 / i:
                selected_line = line
                line_number = i
    return line_number, remove_quotes(selected_line.strip())


def parse_comma_separated_strings(input_string):
    # Split the string by comma and strip whitespace from each element
    return [element.strip() for element in input_string.split(',')]


def read_file_to_list(file_path):
    """
    Reads a text file and returns its contents as a list, with each line as a separate element.

    Args:
    file_path (str): Path to the text file.

    Returns:
    list: A list containing each line of the file as a separate element.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Removing any trailing newline characters from each line
    lines = [line.strip().lower() for line in lines]

    return lines
