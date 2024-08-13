#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define miscellaneous functions for SOSSISSE

Created on 2024-08-13 at 11:23

@author: cook

Rule: import onlys from sossisse.core.base
"""
import os
import string
from datetime import datetime
from hashlib import blake2b

import numpy as np

from sossisse.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.misc'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get all chars
CHARS = string.ascii_uppercase + string.digits


# =============================================================================
# Define printing functions
# =============================================================================
def color(message, colorname):
    colors = dict()
    colors['BLACK'] = '\033[90;1m'
    colors['RED'] = '\033[1;91;1m'
    colors['GREEN'] = '\033[92;1m'
    colors['YELLOW'] = '\033[1;93;1m'
    colors['BLUE'] = '\033[94;1m'
    colors['MAGENTA'] = '\033[1;95;1m'
    colors['CYAN'] = '\033[1;96;1m'
    colors['WHITE'] = '\033[97;1m'
    colors['ENDC'] = '\033[0;0m'

    return colors[colorname.upper()] + message + colors['ENDC']


def sossart():
    """
    Print the SOSS ART logo

    :return: None: prints to screen
    """
    # Etienne needs his sausage picture
    v = """   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⠷⠶⠖⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣾⣿⣶⣄⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⢿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣶⣶⣤⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠻⠿⢿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⣿⣷⣶⣤⣾⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⣠⡀⠲⢿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣴⣿⣿⣿⣶⣤⣄⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠠⢤⣴⡆⠈⠙⠛⠛⠛⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠈⠀⠀⠀                       """
    # we split this up to add color and a border
    v = v.split('\n')
    # add the color
    for i in range(len(v)):
        v[i] = color(v[i], 'red')
    # add the border
    for i in range(len(v)):
        v[i] = color('│', 'white') + v[i] + color('│', 'white')
    # join it all back up
    v = '\n'.join(v)
    # add a top and bottom border
    v = '\n\n' + color('┌──────────────────────────────────┐', 'white') + \
        '\n' + v + '\n' + color('└──────────────────────────────────┘', 'white')
    # add space around the picture
    vlen = np.max([len(lenght_v) for lenght_v in v.split('\n')])
    w = os.get_terminal_size().columns
    dw = (w - vlen // 2) // 2
    # add the white space
    v = v.replace('\n', '\n' + ' ' * dw)
    # print to screen
    print(v)


def art(word: str, color1: str = 'magenta', color2: str = 'red'):
    """
    Print a word in ascii art

    :param word: str, the word to print
    :param color1: str, the color of the border
    :param color2: str, the color of the word
    :return:
    """
    # letters
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y', 'Z']
    # length of each letter in the word art array
    length = [3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3,
              3, 4, 3, 4, 3, 3]
    # lower case letters
    low1 = "┌─┐┌┐ ┌─┐┌┬┐┌─┐┌─┐┌─┐┬ ┬┬  ┬┬┌─┬  ┌┬┐┌┐┌┌─┐┌─┐┌─┐ ┬─┐┌─┐┌┬┐┬ ┬┬  ┬┬ ┬─┐ ┬┬ ┬┌─┐"
    low2 = "├─┤├┴┐│   ││├┤ ├┤ │ ┬├─┤│  │├┴┐│  │││││││ │├─┘│─┼┐├┬┘└─┐ │ │ │└┐┌┘│││┌┴┬┘└┬┘┌─┘"
    low3 = "┴ ┴└─┘└─┘─┴┘└─┘└  └─┘┴ ┴┴└─┘┴ ┴┴─┘┴ ┴┘└┘└─┘┴  └─┘└┴└─└─┘ ┴ └─┘ └┘ └┴┘┴ └─ ┴ └─┘"
    # upper case letters
    up1 = "╔═╗╔╗ ╔═╗╔╦╗╔═╗╔═╗╔═╗╦ ╦╦  ╦╦╔═╦  ╔╦╗╔╗╔╔═╗╔═╗╔═╗ ╦═╗╔═╗╔╦╗╦ ╦╦  ╦╦ ╦═╗ ╦╦ ╦╔═╗"
    up2 = "╠═╣╠╩╗║   ║║║╣ ╠╣ ║ ╦╠═╣║  ║╠╩╗║  ║║║║║║║ ║╠═╝║═╬╗╠╦╝╚═╗ ║ ║ ║╚╗╔╝║║║╔╩╦╝╚╦╝╔═╝"
    up3 = "╩ ╩╚═╝╚═╝═╩╝╚═╝╚  ╚═╝╩ ╩╩╚═╝╩ ╩╩═╝╩ ╩╝╚╝╚═╝╩  ╚═╝╚╩╚═╚═╝ ╩ ╚═╝ ╚╝ ╚╩╝╩ ╚═ ╩ ╚═╝"
    # add other characters
    letters = letters + ['-', ' ', '.', '0', '1', '2', '3', '4', '5', '6',
                         '7', '8', '9', '[', ']', '?', '!']
    # add length of other characters
    length = length + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # add lower case letters
    low1 = low1 + "         ┌─┐ ┐ ┌─┐┌─┐┌ ┐┌─┐┌─┐┌─┐┌─┐┌─┐ ┌  ┐ ┌─┐ ┐ "
    low2 = low2 + "───      │ │ │ ┌─┘ ─┤└─┤└─┐├─┐  │├─┤└─┤ │  │  ┌┘ │ "
    low3 = low3 + "      ·  └─┘─┴─└─┘└─┘  ┘└─┘└─┘  ┴└─┘└─┘ └  ┘  o  o "
    # start the word art strings
    low_1 = ""
    low_2 = ""
    low_3 = ""
    # convert word to lower case
    letter = np.array([letter.lower() for letter in letters])
    # find the start and end of each letter in the word
    l1 = np.array([(np.cumsum(length))[ll.lower() == letter][0] for ll in word])
    l2 = np.array([np.array(length)[ll.lower() == letter][0] for ll in word])
    # find the difference between the start and end of each letter
    l0 = l1 - l2
    # loop around each letter in the word
    for it in range(len(l1)):
        if word[it] == word[it].lower():
            low_1 += low1[l0[it]:l1[it]]
            low_2 += low2[l0[it]:l1[it]]
            low_3 += low3[l0[it]:l1[it]]
        else:
            low_1 += up1[l0[it]:l1[it]]
            low_2 += up2[l0[it]:l1[it]]
            low_3 += up3[l0[it]:l1[it]]
    # add border
    low_0 = color('╔' + '═' * (len(low_1) + 2) + '╗', color1)
    low_4 = color('╚' + '═' * (len(low_1) + 2) + '╝', color1)
    low_1 = color('║ ', color1) + color(low_1, color2) + color(' ║', color1)
    low_2 = color('║ ', color1) + color(low_2, color2) + color(' ║', color1)
    low_3 = color('║ ', color1) + color(low_3, color2) + color(' ║', color1)
    # add spaces
    w = os.get_terminal_size().columns
    dw = (w - len(low_1) // 2) // 2
    low_0 = ' ' * dw + low_0
    low_1 = ' ' * dw + low_1
    low_2 = ' ' * dw + low_2
    low_3 = ' ' * dw + low_3
    low_4 = ' ' * dw + low_4
    # return the combined string
    string_list = [low_0, low_1, low_2, low_3, low_4]
    return '\n' + '\n'.join(string_list) + '\n'


def printc(message: str, msg_type: str, print_time: bool = True):
    """
    Print a message with color

    :param message: str, the message to print
    :param msg_type: str, the type of message (info, bad1, bad2, bad3, number)
    :param print_time: bool, if True print the timestamp

    :return: None, prints to screen
    """
    msg_color = "white"

    if msg_type == 'info':
        msg_color = 'green'

    if msg_type == 'bad1':
        msg_color = 'yellow'

    if msg_type == 'bad2':
        msg_color = 'red'

    if msg_type == 'bad3':
        msg_color = 'magenta'

    if msg_type == 'number':
        msg_color = 'blue'

    if print_time:
        time = datetime.now().strftime('%H:%M:%S.%f')[:-4] + '│'
    else:
        time = ''
    print(color(time + message, msg_color))


def sossice_unique_id(param_file: str) -> str:
    """
    Generate a uniuqe
    :param param_file:
    :return:
    """
    # read yaml file as string
    with open(param_file, 'r') as f:
        param_file_str = f.read()
    # get the unique id line
    return generate_hash(param_file_str, size=5)


def generate_hash(string_text: str, size: int = 10) -> str:
    """
    Generate a hash code based on the 'string_text' of length 'size'

    :param string_text: str, the string to generate hash code from
    :param size: int, the size of the hash to create

    :return: str, the generated hash code
    """
    # need to encode string
    encoded = string_text.encode('utf')
    # we want a hash of 10 characters
    digest = blake2b(encoded, digest_size=size)
    # create hash
    hashstr = digest.hexdigest()
    # return hash
    return str(hashstr)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
