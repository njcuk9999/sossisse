#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define miscellaneous functions for SOSSISSE

Created on 2024-08-13 at 11:23

@author: cook

Rule: import onlys from sossisse.core.base
"""
import os
import sys
import random
import string
import time
import getpass
import pwd
from typing import Any, List, Optional, Tuple

import numpy as np
from astropy.time import Time

from aperocore.core import drs_log
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
WLOG = drs_log.wlog
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


def safe_getuser():
    try:
        return getpass.getuser()
    except Exception:
        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


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
    # get username
    username = safe_getuser()
    # special logo for Etienne only
    if username in ['eartigau', 'spirou']:
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
    title = art('SOSSISSE', color1='white')
    # print to screen
    print(title)
    # version string
    vstring = f'SOSSISSE: v{__version__} - {__date__}'
    # print date and version
    header_line()
    WLOG(None, 'info', vstring, colour='magenta', no_timestamp=True)
    header_line()


def modeart(mode: str):
    """
    Print the instrument+mode

    :param mode: str, the instrument + mode to print

    :return: None, prints to stdout
    """
    # print date and version
    mode_title = art(mode, color1='white')
    print(mode_title)


def art(word: str, color1: str = 'magenta', color2: str = 'red',
        message: Optional[str] = None):
    """
    Print a word in ascii art

    :param word: str, the word to print
    :param color1: str, the color of the border
    :param color2: str, the color of the word
    :param message: optional str, a message to print beneath the word art
    :return: str, the banner with whitespace rendered as spaces and
             unsupported characters rendered as question marks
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
    letters.append('+')
    length.append(3)
    low1 += ' │ '
    low2 += '─┼─'
    low3 += ' │ '
    supported_letters = {character.lower() for character in letters}
    word = ''.join(' ' if character.isspace() else character
                   if character.lower() in supported_letters else '?'
                   for character in word)
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
    if message:
        message = ' '.join(message.splitlines()).strip()
    line_width = max(len(low_1), len(message or ''))
    rule = '═' * (line_width + 2)

    def framed(text: str) -> str:
        return (color('║ ', color1) + color(text.ljust(line_width), color2)
                + color(' ║', color1))

    rows = [color('╔' + rule + '╗', color1),
            framed(low_1), framed(low_2), framed(low_3)]
    if message:
        rows.extend([color('╠' + rule + '╣', color1), framed(message)])
    rows.append(color('╚' + rule + '╝', color1))
    indent = ' ' * ((80 - len(rows[1]) // 2) // 2)
    return '\n' + '\n'.join(indent + row for row in rows) + '\n'


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
              options: List[Any] = None, argname: str = None):
    """
    Ask the user for an input

    :param parameter:
    :param dtype: str, the dtype (str, int, float, bool, path, dir)
    :param options: if set user input must be one of these

    :return:
    """
    # deal with having arguments already
    if argname is not None:
        for argstr in sys.argv:
            if f'--{argname}' in argstr:
                return None
            if f'--help' in argstr:
                return None

    if comment is None:
        param_name = parameter
    else:
        param_name = f'{comment} [{parameter}]'
    # loop around until we get a valid input
    while True:
        prompt = f'\nPlease input {param_name}\n\tDtype: {dtype}'
        # add options if they are given
        if options is not None:
            prompt += (f'\n\tOptions:')
            for it, option in enumerate(options):
                prompt += f'\n\t - {it+1}: {option}'
        # get user input
        user_input = input(prompt + '\n >> ')
        # ----------------------------------------------------------------------
        if options is not None:
            # if user input is an interger we can try to get it from the options
            options_int = [str(i + 1) for i in range(len(options))]
            # get the option
            if user_input in options_int:
                user_input = options[int(user_input) - 1]
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


def header_line(params=None):
    WLOG(params, 'info', '*' * base.CONSOLE_WIDTH, colour='magenta',
         no_timestamp=True, wrap=False)


def end_recipe(params=None):
    header_line(params)
    WLOG(params, 'info', 'Code finished successfully', colour='magenta')
    header_line(params)

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
