import sys
import os
from typing import List, Optional, Any, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sample_dir = os.path.join(project_root, "data", "sample_submission", "sample_submission")
if sample_dir not in sys.path:
    sys.path.insert(0, sample_dir)

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
)

# Load card and attack counts from official cg API
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1

num_words_encoder = 24
encoder_size = 50000

decoder_main_feature = 8
decoder_attack_offset = 14
decoder_card_offset = decoder_attack_offset + attack_count
recover_spec_val = int(SelectContext.RECOVER_SPECIAL_CONDITION.value) if hasattr(SelectContext.RECOVER_SPECIAL_CONDITION, "value") else int(SelectContext.RECOVER_SPECIAL_CONDITION)
decoder_size = decoder_card_offset + (1 + decoder_main_feature + recover_spec_val) * card_count + 1000


class SparseVector:
    """Sparse vector builder for EmbeddingBag inputs in PyTorch."""

    def __init__(self):
        self.index: List[int] = []
        self.value: List[float] = []
        self.offset: List[int] = []
        self.pos: int = 0

    def add(self, index: int, value: Union[float, int, bool]):
        val = float(value)
        if val != 0.0:
            self.index.append(self.pos + index)
            self.value.append(val)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: Union[float, int, bool]):
        val = float(value)
        if val != 0.0:
            self.index.append(self.pos)
            self.value.append(val)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


def add_card(sv: SparseVector, card: Union[Card, Pokemon, None]):
    if card is not None and hasattr(card, "id"):
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv: SparseVector, cards: Optional[List[Card]], value: float):
    if cards is not None:
        for card in cards:
            if hasattr(card, "id"):
                sv.add(card.id, value)
    sv.add_pos(card_count)


def add_pokemon(sv: SparseVector, poke: Optional[Pokemon]):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        hp_val = getattr(poke, "hp", 0) / 400.0
        sv.add_single(hp_val)
        add_card(sv, poke)
        add_cards(sv, getattr(poke, "tools", None), 1.0)
        add_cards(sv, getattr(poke, "energyCards", None), 0.5)


def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60.0)
    sv.add_single(len(ps.discard) / 60.0)
    sv.add_single(ps.handCount / 8.0)
    sv.add_single(len(ps.bench) / 5.0)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(getattr(ps, "poisoned", False))
    sv.add_single(getattr(ps, "burned", False))
    sv.add_single(getattr(ps, "asleep", False))
    sv.add_single(getattr(ps, "paralyzed", False))
    sv.add_single(getattr(ps, "confused", False))

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: List[int]) -> SparseVector:
    """Builds SparseVector representation of the entire board state for Encoder input."""
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):  # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if len(ps.active) > 0:
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(card_count)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10.0)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Union[Pokemon, Card, None]:
    """Helper to locate card object based on area type and index."""
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index] if obs.select and obs.select.deck and index < len(obs.select.deck) else None
    elif area == AreaType.HAND:
        return ps.hand[index] if index < len(ps.hand) else None
    elif area == AreaType.DISCARD:
        return ps.discard[index] if index < len(ps.discard) else None
    elif area == AreaType.ACTIVE:
        return ps.active[index] if index < len(ps.active) else None
    elif area == AreaType.BENCH:
        return ps.bench[index] if index < len(ps.bench) else None
    elif area == AreaType.PRIZE:
        return ps.prize[index] if index < len(ps.prize) else None
    elif area == AreaType.STADIUM:
        return obs.current.stadium[index] if index < len(obs.current.stadium) else None
    return None


def decoder_main(sv: SparseVector, feature_index: int, card: Union[Card, Pokemon, None]):
    if card is not None and hasattr(card, "id"):
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv: SparseVector, context_val: int, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context_val) * card_count + card_id, 1)


def decoder_card(sv: SparseVector, context_val: int, card: Union[Card, Pokemon, None]):
    if card is not None and hasattr(card, "id"):
        decoder_card_id(sv, context_val, card.id)


def get_decoder_input(obs: Observation, your_deck: List[int]) -> SparseVector:
    """Builds SparseVector representation of action options for Decoder input."""
    sv = SparseVector()
    
    options = obs.select.option[:64] if obs and obs.select and obs.select.option else []
    your_index = obs.current.yourIndex if obs and obs.current else 0
    ps = obs.current.players[your_index] if obs and obs.current and obs.current.players else None
    raw_ctx = obs.select.context if obs and obs.select else 0
    context_val = raw_ctx.value if hasattr(raw_ctx, "value") else int(raw_ctx)

    for o in options:
        sv.word_start()
        opt_type = o.type.value if hasattr(o.type, "value") else int(o.type)

        if opt_type == int(OptionType.END.value if hasattr(OptionType.END, "value") else OptionType.END):
            sv.add(1, 1)
        elif opt_type == int(OptionType.YES.value if hasattr(OptionType.YES, "value") else OptionType.YES):
            sv.add(2, 1)
        elif opt_type == int(OptionType.NO.value if hasattr(OptionType.NO, "value") else OptionType.NO):
            sv.add(3, 1)
        elif opt_type == int(OptionType.SPECIAL_CONDITION.value if hasattr(OptionType.SPECIAL_CONDITION, "value") else OptionType.SPECIAL_CONDITION):
            spec_val = getattr(o, "specialConditionType", 0)
            spec_val = spec_val.value if hasattr(spec_val, "value") else int(spec_val)
            sv.add(4 + spec_val, 1)
        elif opt_type == int(OptionType.NUMBER.value if hasattr(OptionType.NUMBER, "value") else OptionType.NUMBER):
            sv.add(9 + min(getattr(o, "number", 0), 4), 1)
        elif opt_type == int(OptionType.ATTACK.value if hasattr(OptionType.ATTACK, "value") else OptionType.ATTACK):
            atk_id = getattr(o, "attackId", 0)
            sv.add(decoder_attack_offset + atk_id, 1)
        elif opt_type == int(OptionType.PLAY.value if hasattr(OptionType.PLAY, "value") else OptionType.PLAY):
            o_idx = getattr(o, "index", 0)
            card_obj = ps.hand[o_idx] if ps and hasattr(ps, "hand") and o_idx < len(ps.hand) else None
            decoder_main(sv, 0, card_obj)
        elif opt_type == int(OptionType.ATTACH.value if hasattr(OptionType.ATTACH, "value") else OptionType.ATTACH):
            decoder_main(sv, 1, get_card(obs, getattr(o, "area", AreaType.HAND), getattr(o, "index", 0), your_index))
            decoder_main(sv, 2, get_card(obs, getattr(o, "inPlayArea", AreaType.ACTIVE), getattr(o, "inPlayIndex", 0), your_index))
        elif opt_type == int(OptionType.EVOLVE.value if hasattr(OptionType.EVOLVE, "value") else OptionType.EVOLVE):
            decoder_main(sv, 3, get_card(obs, getattr(o, "area", AreaType.HAND), getattr(o, "index", 0), your_index))
            decoder_main(sv, 4, get_card(obs, getattr(o, "inPlayArea", AreaType.ACTIVE), getattr(o, "inPlayIndex", 0), your_index))
        elif opt_type == int(OptionType.ABILITY.value if hasattr(OptionType.ABILITY, "value") else OptionType.ABILITY):
            decoder_main(sv, 5, get_card(obs, getattr(o, "area", AreaType.ACTIVE), getattr(o, "index", 0), your_index))
        elif opt_type == int(OptionType.DISCARD.value if hasattr(OptionType.DISCARD, "value") else OptionType.DISCARD):
            decoder_main(sv, 6, get_card(obs, getattr(o, "area", AreaType.HAND), getattr(o, "index", 0), your_index))
        elif opt_type == int(OptionType.RETREAT.value if hasattr(OptionType.RETREAT, "value") else OptionType.RETREAT):
            decoder_main(sv, 7, ps.active[0] if ps and hasattr(ps, "active") and len(ps.active) > 0 else None)
        elif opt_type == int(OptionType.CARD.value if hasattr(OptionType.CARD, "value") else OptionType.CARD):
            p_idx = getattr(o, "playerIndex", your_index)
            decoder_card(sv, context_val, get_card(obs, getattr(o, "area", AreaType.HAND), getattr(o, "index", 0), p_idx))
        elif opt_type == int(OptionType.TOOL_CARD.value if hasattr(OptionType.TOOL_CARD, "value") else OptionType.TOOL_CARD):
            p_idx = getattr(o, "playerIndex", your_index)
            card_obj = get_card(obs, getattr(o, "area", AreaType.ACTIVE), getattr(o, "index", 0), p_idx)
            tool_idx = getattr(o, "toolIndex", 0)
            if card_obj and hasattr(card_obj, "tools") and card_obj.tools and tool_idx < len(card_obj.tools):
                decoder_card(sv, context_val, card_obj.tools[tool_idx])
        elif opt_type in (int(OptionType.ENERGY_CARD.value if hasattr(OptionType.ENERGY_CARD, "value") else OptionType.ENERGY_CARD), int(OptionType.ENERGY.value if hasattr(OptionType.ENERGY, "value") else OptionType.ENERGY)):
            p_idx = getattr(o, "playerIndex", your_index)
            card_obj = get_card(obs, getattr(o, "area", AreaType.ACTIVE), getattr(o, "index", 0), p_idx)
            e_idx = getattr(o, "energyIndex", 0)
            if card_obj and hasattr(card_obj, "energyCards") and card_obj.energyCards and e_idx < len(card_obj.energyCards):
                decoder_card(sv, context_val, card_obj.energyCards[e_idx])
        elif opt_type == int(OptionType.SKILL.value if hasattr(OptionType.SKILL, "value") else OptionType.SKILL):
            decoder_card_id(sv, context_val, getattr(o, "cardId", 0))

    # Always pad remaining option words up to 64
    for _ in range(64 - len(options)):
        sv.word_start()
        sv.add(0, 0.0)

    return sv


class StateEncoder:
    """Wrapper class providing encode API for PTCGObservation."""

    def __init__(self):
        self.num_words_encoder = num_words_encoder
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size

    def encode(self, obs: Observation, your_deck: List[int] = None) -> SparseVector:
        if your_deck is None:
            your_deck = []
        return get_encoder_input(obs, your_deck)

    def encode_decoder(self, obs: Observation, your_deck: List[int] = None) -> SparseVector:
        if your_deck is None:
            your_deck = []
        return get_decoder_input(obs, your_deck)
