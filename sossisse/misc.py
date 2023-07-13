import os
from datetime import datetime
import numpy as np


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
    v = v.split('\n')
    for i in range(len(v)):
        v[i] = color(v[i], 'red')
    for i in range(len(v)):
        v[i] = color('│', 'white') + v[i] + color('│', 'white')
    v = '\n'.join(v)
    v = '\n\n' + color('┌──────────────────────────────────┐', 'white') + \
        '\n' + v + '\n' + color('└──────────────────────────────────┘', 'white')

    vlen = np.max([len(lenght_v) for lenght_v in v.split('\n')])
    w = os.get_terminal_size().columns
    dw = (w - vlen // 2) // 2

    v = v.replace('\n', '\n' + ' ' * dw)
    print(v)

    return


def art(word, color1='MAGENTA', color2='red'):
    letters = \
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z']
    length = \
        [3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 4, 3, 3]

    low1 = "┌─┐┌┐ ┌─┐┌┬┐┌─┐┌─┐┌─┐┬ ┬┬  ┬┬┌─┬  ┌┬┐┌┐┌┌─┐┌─┐┌─┐ ┬─┐┌─┐┌┬┐┬ ┬┬  ┬┬ ┬─┐ ┬┬ ┬┌─┐"
    low2 = "├─┤├┴┐│   ││├┤ ├┤ │ ┬├─┤│  │├┴┐│  │││││││ │├─┘│─┼┐├┬┘└─┐ │ │ │└┐┌┘│││┌┴┬┘└┬┘┌─┘"
    low3 = "┴ ┴└─┘└─┘─┴┘└─┘└  └─┘┴ ┴┴└─┘┴ ┴┴─┘┴ ┴┘└┘└─┘┴  └─┘└┴└─└─┘ ┴ └─┘ └┘ └┴┘┴ └─ ┴ └─┘"
    up1 = "╔═╗╔╗ ╔═╗╔╦╗╔═╗╔═╗╔═╗╦ ╦╦  ╦╦╔═╦  ╔╦╗╔╗╔╔═╗╔═╗╔═╗ ╦═╗╔═╗╔╦╗╦ ╦╦  ╦╦ ╦═╗ ╦╦ ╦╔═╗"
    up2 = "╠═╣╠╩╗║   ║║║╣ ╠╣ ║ ╦╠═╣║  ║╠╩╗║  ║║║║║║║ ║╠═╝║═╬╗╠╦╝╚═╗ ║ ║ ║╚╗╔╝║║║╔╩╦╝╚╦╝╔═╝"
    up3 = "╩ ╩╚═╝╚═╝═╩╝╚═╝╚  ╚═╝╩ ╩╩╚═╝╩ ╩╩═╝╩ ╩╝╚╝╚═╝╩  ╚═╝╚╩╚═╚═╝ ╩ ╚═╝ ╚╝ ╚╩╝╩ ╚═ ╩ ╚═╝"

    letters = letters + ['-', ' ', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '?', '!']
    length = length + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    low1 = low1 + "         ┌─┐ ┐ ┌─┐┌─┐┌ ┐┌─┐┌─┐┌─┐┌─┐┌─┐ ┌  ┐ ┌─┐ ┐ "
    low2 = low2 + "───      │ │ │ ┌─┘ ─┤└─┤└─┐├─┐  │├─┤└─┤ │  │  ┌┘ │ "
    low3 = low3 + "      ·  └─┘─┴─└─┘└─┘  ┘└─┘└─┘  ┴└─┘└─┘ └  ┘  o  o "

    low_1 = ""
    low_2 = ""
    low_3 = ""

    letter = np.array([letter.lower() for letter in letters])

    l1 = np.array([(np.cumsum(length))[ll.lower() == letter][0] for ll in word])
    l2 = np.array([np.array(length)[ll.lower() == letter][0] for ll in word])
    l0 = l1 - l2

    for i in range(len(l1)):
        if word[i] == word[i].lower():
            low_1 += low1[l0[i]:l1[i]]
            low_2 += low2[l0[i]:l1[i]]
            low_3 += low3[l0[i]:l1[i]]
        else:
            low_1 += up1[l0[i]:l1[i]]
            low_2 += up2[l0[i]:l1[i]]
            low_3 += up3[l0[i]:l1[i]]

    low_0 = color('╔' + '═' * (len(low_1) + 2) + '╗', color1)
    low_4 = color('╚' + '═' * (len(low_1) + 2) + '╝', color1)

    low_1 = color('║ ', color1) + color(low_1, color2) + color(' ║', color1)
    low_2 = color('║ ', color1) + color(low_2, color2) + color(' ║', color1)
    low_3 = color('║ ', color1) + color(low_3, color2) + color(' ║', color1)

    w = os.get_terminal_size().columns
    dw = (w - len(low_1) // 2) // 2
    low_0 = ' ' * dw + low_0
    low_1 = ' ' * dw + low_1
    low_2 = ' ' * dw + low_2
    low_3 = ' ' * dw + low_3
    low_4 = ' ' * dw + low_4

    return '\n' + low_0 + '\n' + low_1 + '\n' + low_2 + '\n' + low_3 + '\n' + low_4 + '\n'


def printc(message, msg_type, print_time=True):
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
