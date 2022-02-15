#%%
import torch
import numpy as np
import random
#%% make the initial deck

scores = [0,0]

total_cards = 60
start_cards = 8

colors = ['green','white','blue','red','yellow']
cards = [2,3,4,5,6,7,8,9,10,'b','b','b']

card_deck = []
discard_deck = []
expeditions = []
for color in colors:
    for card in cards:
        card_deck.append({'color': color, 'val': card})

#%% give both players 8 random cards

hands = [[],[]]

for card_idx in range(start_cards):
    hands[0].append(card_deck.pop(random.randrange(len(card_deck))))
    hands[1].append(card_deck.pop(random.randrange(len(card_deck))))

#%%

def can_build(hand,expedition):
    


#%%
play_actions = ['build','discard']
draw_actions = ['draw_blind','draw_discard']

player = 0

while(len(card_deck)>0):

    #can we build?
    if can_build(hands[player],expeditions[player]):
        action = play_actions[random.randint(0,1)]
    else:
        action = 'discard'

    #if no cards on discard deck, draw from blind deck
    if len(discard_deck) == 0:
        draw_action = 'draw_blind'
    else:
        draw_action = draw_actions[random.randint(0,1)]

    










