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
play_actions = ['build','discard']
draw_actions = ['draw_blind','draw_discard']

#%%
class Expedition:

    def __init__(self):
        self.expeditions = {
            'green' : [],
            'blue' : [],
            'yellow' : [],
            'red' : [],
            'white' : []
        }

    def add_card(self,card):
        self.expeditions[card['color']].append(card['val'])

    def can_build(self, cards):

        
        for card in cards:
            card_color = card['color']
            card_val = card['val']
            #check if color in hand is not yet started
            if len(self.expeditions[card_color]) == 0:
                return True

            #check if val in hand is boost and a boost is on top
            elif card_val == 'b':
                if self.expeditions[card_color][-1] == 'b':
                    return True

            #check if val in hand is larger than val of started exp
            elif card_val > self.expeditions[card_color][-1]:
                return True

            else:
                return False


#%%

exps = [Expedition(),Expedition()]Lost

#%%
player = 0
while(len(card_deck)>0):

    exp = exps[player]



    #can we build?
    if exp.can_build(hands[player]):
        action = play_actions[random.randint(0,1)]
    else:
        action = 'discard'

    #if no cards on discard deck, draw from blind deck
    if len(discard_deck) == 0:
        draw_action = 'draw_blind'
    else:
        draw_action = draw_actions[random.randint(0,1)]
#%%

#%%
exp = Expedition()
exp.add_card(card_deck[0])


exp.can_build(hands[0])
#%%













