#%%
import torch
import numpy as np
import random
from classes import Hand, Expedition, Pile
#%% make the initial deck and shuffle it

scores = [0,0]

total_cards = 60
start_cards = 8

colors = ['green','white','blue','red','yellow']
cards = [2,3,4,5,6,7,8,9,10,'b','b','b']

card_deck = []
discard_deck = []
expeditions = []
id = 1
for color in colors:
    for card in cards:
        card_deck.append({'id': id,'color': color, 'val': card})
        id +=1

random.shuffle(card_deck)

#%%
hands = [Hand(),Hand()]
exps = [Expedition(),Expedition()]
centre_pile = Pile()
play_actions = ['build','discard']
draw_actions = ['draw_blind','draw_discard']
#%% give both players 8 random cards


for card_idx in range(start_cards):
    hands[0].add_card(card_deck.pop())
    hands[1].add_card(card_deck.pop())


#%%
player = 0
while(len(card_deck)>0):
    print(f'cards in deck {len(card_deck)}')
    print(f'player {player}')

    exp = exps[player]
    hand = hands[player]

    # can we build?
    possible_builds = exp.get_possible_builds(hand.cards)
    if len(possible_builds) > 0:
        action = play_actions[random.randint(0,1)]
    else:
        action = 'discard'

    if action == 'build':
        card_id_played = random.randint(0,len(possible_builds)-1)
        played_card = possible_builds[card_id_played]
        exp.add_card(played_card) # add card to expedition
        hand.remove_card(played_card) # remove card from hand
    
    elif action == 'discard':
        card_id_played = random.randint(0,len(hand.cards)-1)
        played_card = hand.cards[card_id_played]
        centre_pile.add_card(played_card) # add card to centre pile
        hand.remove_card(played_card) # remove card from hand


    # if no cards on discard deck, draw from blind deck
    visible_cards = centre_pile.get_visible_cards()
    if len(visible_cards) == 0:
        draw_action = 'draw_blind'
    else:
        draw_action = draw_actions[random.randint(0,1)]

    if draw_action == 'draw_blind':
        new_card = card_deck.pop()
        hand.add_card(new_card)

    elif draw_action == 'draw_discard':
        card_id_draw = random.randint(0,len(visible_cards)-1)
        new_card = visible_cards[card_id_draw]
        centre_pile.draw_card(new_card) # add card to centre pile
        hand.add_card(played_card) # remove card from hand       
        
    if player == 0:
        player = 1
    elif player == 1:
        player = 0
