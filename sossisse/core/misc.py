#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define miscellaneous functions for SOSSISSE

Created on 2024-08-13 at 11:23

@author: cook

Rule: import onlys from sossisse.core.base
"""
import os
import random
import string
from datetime import datetime
import time
from typing import Any, List, Tuple

import numpy as np
from astropy.time import Time

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
# set the log level
LOG_LEVEL = 'INFO'
# log levels
LOG_LEVELS = dict()
LOG_LEVELS['DEBUG'] = 0
LOG_LEVELS['SETUP'] = 0.5
LOG_LEVELS['INFO'] = 1
LOG_LEVELS['WARNING'] = 2
LOG_LEVELS['ERROR'] = 3
LOG_LEVELS['ALERT'] = 4
LOG_LEVELS['NUMBER'] = 1
# flag for sossart printing
SOSSART_USED = False


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
    global SOSSART_USED
    # if we have already printed SOSSART then return and don't print again
    if SOSSART_USED:
        return
    # set the flag
    SOSSART_USED = True
    # special logo for Etienne only
    if os.getlogin() in ['eartigau', 'spirou']:
        # Etienne needs his sausage picture
        v = """                                                                                    
                                                        ,.  @@@@ ,@@.               
                                                       #@(/(///((/((@               
                                                        %&/(*((*((#@                
                                                       /@@@#((((#@@@                
                                                    @@(/********(//((%@.            
                                                  @(/*****//(///(////(((@           
                                                &&//***((////////////((((@*         
                                               &@(/***(////////////////(((@.        
                                               @///*///////////////////(((#@        
                                              (@(((/(/////////////////(((((@        
                                              @%(//////////////////////((((@.       
                                              @%(//////////////////////((((@,       
                                              @#(/////////////////////(((((@        
                                              @(/(/////////////////////((((@        
                                              @#/////////////////////(/(((#@        
                                              @///////////////////////((((@,        
                                             ,@(//////////////////////(((#@         
                                             @/////////////////////(/((((@.         
                                           .@//(////////////////////((((%@          
                                          @&(/(////////////////////((@@@@           
                                        ,@////////////////////////(((((@(           
                                      .@(////////////////////////(((((%@            
                                    #@((////////////////////////(((((#@             
                                  @&(/**/(////////////////////((%&(((@              
                              .@&//****/(/////////////////////(((((@@               
                           @@#(*****////////////////////////((((((&@                
                        @@(/*****/(//////////////////////(/((((((@,                 
                     ,@((*****/(////////////////////////(((((((@@                   
                    @#/****////////////////////////////(((((@@@                     
                   @(/***/(////////////////////////(/((((((@@                       
           *@/(#@@@@(//*///////////////////////(//(((((((@%                         
            (@(/**/@//(/**(//////////////////(/(((&#((@@                            
            (@/(///@%//(((////////////////((((((((#@&                               
              @@//((%@(/(////////////((((&((((#@@.                                  
              @&/((%@  @@#(/(///(((((((((&@@/                                       
               @@@@          &@@@@@@&.                                              
                                                                                    """
        # we split this up to add color and a border
        v = v.split('\n')
        llen = len(v[0])
        # add the color
        for i in range(len(v)):
            v[i] = color(v[i], 'red')
        # add the border
        for i in range(len(v)):
            v[i] = color('│', 'white') + v[i] + color('│', 'white')
        # join it all back up
        v = '\n'.join(v)
        # add a top and bottom border
        bordertop = '┌' + '─' * llen + '┐'
        borderbottom = '└' + '─' * llen + '┘'
        v = ('\n\n' + color(bordertop, 'white') +
             '\n' + v + '\n' + color(borderbottom, 'white'))
        # add space around the picture
        vlen = np.max([len(lenght_v) for lenght_v in v.split('\n')])
        try:
            w = os.get_terminal_size().columns
        except OSError:
            w = 80
        dw = (w - vlen // 2) // 2
        # add the white space
        v = v.replace('\n', '\n' + ' ' * dw)
    # otherwise we print the normal SOSSISSE logo
    v = art('SOSSISSE', color1='white')
    # print to screen
    print(v)
    # version string
    vstring = f'SOSSISSE: v{__version__} - {__date__}'
    # print date and version
    printc(vstring, 'ALERT', print_time=False)


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
    # try:
    #     w = os.get_terminal_size().columns
    # except OSError:
    w = 80
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
    # formally bad1
    if msg_type.lower() in 'warning':
        msg_color = 'yellow'
    # formally bad2
    elif msg_type.lower() in 'error':
        msg_color = 'red'
    # formally bad3
    elif msg_type.lower() == 'alert':
        msg_color = 'magenta'

    elif msg_type.lower() == 'number':
        msg_color = 'blue'
    else:
        msg_color = 'green'
    # get the time message
    if print_time:
        time = datetime.now().strftime('%H:%M:%S.%f')[:-4] + '│'
    else:
        time = ''
    # at this point we set everything else at the info level
    if msg_type.upper() not in LOG_LEVELS:
        msg_type = 'INFO'
    # print if the log level is high enough
    if LOG_LEVELS[msg_type.upper()] >= LOG_LEVELS[LOG_LEVEL.upper()]:
        print(color(time + message, msg_color), flush=True)


def sossice_unique_id(param_file: str) -> str:
    """
    Assign a sossisse id based on the time now and return it and the
    time now

    :return: the process id and the human time at creation
    :rtype: Tuple[str, str]
    """
    # get unix char code
    unixtime, humantime, rval = unix_char_code()
    # write sid
    sid = 'SID-{0:020d}-{1}'.format(int(unixtime), rval)
    # return sid
    return sid


def unix_char_code() -> Tuple[float, str, str]:
    """
    Get the time now (using astropy.Time) and return the unix time
    human time and a random code of 4 characters

    :return: tuple, 1. the unix time now, 2. the human time now, 3. a random
             set of 4 characters
    """
    # set function name
    # _ = display_func('unix_char_code', __NAME__)
    # we need a random seed
    np.random.seed(random.randint(1, 2 ** 30))
    # generate a random number (in case time is too similar)
    #  -- happens a lot in multiprocessing
    rint = np.random.randint(1000, 9999, 1)[0] / 1e7
    # wait a fraction of time (between 1us and 1ms)
    time.sleep(float(rint))
    # get the time now from astropy
    timenow = Time.now()
    # get unix and human time from astropy time now
    unixtime = timenow.unix * 1e7
    humantime = timenow.iso
    # generate random four characters to make sure sid is unique
    rval = ''.join(np.random.choice(list(CHARS), size=4))
    return unixtime, humantime, rval


def get_input(parameter, dtype: str = 'str', comment: str = None,
              options: List[Any] = None):
    """
    Ask the user for an input

    :param parameter:
    :param dtype: str, the dtype (str, int, float, bool, path, dir)
    :param options: if set user input must be one of these

    :return:
    """
    if comment is None:
        param_name = parameter
    else:
        param_name = f'{comment} [{parameter}]'
    # loop around until we get a valid input
    while True:
        prompt = f'\nPlease input {param_name}\n\tDtype: {dtype}'
        # add options if they are given
        if options is not None:
            prompt += (f'\n\tOptions: {options}')
            for option in options:
                prompt += f'\n\t - {option}'
        # get user input
        user_input = input(prompt + '\n')
        # ----------------------------------------------------------------------
        if dtype == 'str':
            value = str(user_input).strip()
        elif dtype == 'int':
            try:
                value = int(user_input)
            except ValueError:
                print(f'Error: Input must be an integer')
                continue
        elif dtype == 'float':
            try:
                value = float(user_input)
            except ValueError:
                print(f'Error: Input must be a float')
                continue
        elif dtype == 'bool':
            if user_input.lower() in ['true', 't', 'yes', 'y', '1']:
                value = True
            elif user_input.lower() in ['false', 'f', 'no', 'n', '0']:
                value = False
            else:
                print(f'Error: Input must be a boolean')
                continue
        elif dtype == 'path':
            if os.path.exists(user_input):
                value = user_input
            else:
                print(f'Error: Path does not exist')
                continue
        elif dtype == 'dir':
            # clean up name
            user_input = user_input.strip(' ')
            user_input = user_input.replace('~', os.path.expanduser('~'))
            # get directory
            directory = os.path.dirname(user_input)

            if os.path.exists(directory):
                value = user_input
            else:
                print(f'Error: {directory} does not exist, please change '
                      f'or create.')
                continue
        else:
            value = user_input
        # ----------------------------------------------------------------------
        # check options
        if options is not None:
            if value not in options:
                print(f'Error: Input must be one of {options}')
                continue
        # ----------------------------------------------------------------------
        return value


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
