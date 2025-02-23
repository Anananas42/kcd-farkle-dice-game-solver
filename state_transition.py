from itertools import combinations, combinations_with_replacement
from collections import Counter
from typing import List, Tuple, Optional, Dict

DiceValues = Tuple[int, ...]  # Tuple of 1-6 integers
ScoringCombination = Tuple[List[int], int]  # (dice_list, score)
CombinationResult = Optional[ScoringCombination]  # None or (dice_list, score)
LookupTable = Dict[DiceValues, List[CombinationResult]]

class StateTransition:
    def __init__(self, sorted_dice: List[int],  n_set_aside: int):
        self.n_set_aside = n_set_aside
        self.value = self.getBestCombinationValue(sorted_dice, n_set_aside)

    def getBestCombinationValue(sorted_dice: List[int],  n_set_aside: int) -> int:
        return None
    
    @staticmethod
    def generate_dice_combinations_lookup() -> LookupTable:
        """
        Generates a lookup table for all possible sorted combinations of 1-6 dice
        using the getBestCombinationsForUpToNAside method.
        
        Returns:
            Dict[Tuple[int, ...], List[Optional[Tuple[List[int], int]]]]: A dictionary where:
                - key: tuple of sorted dice values (1-6 numbers)
                - value: list of length 6 containing possible combinations and their scores,
                        where each element is either None or a tuple of (dice_list, score)
        """
        lookup_table: LookupTable = {}
        dice_values: range = range(1, 7)  # 1-6
        
        # Generate all possible lengths of dice combinations (1-6 dice)
        for length in range(1, 7):
            # Generate all possible combinations of that length
            # combinations_with_replacement gives us sorted combinations by default
            for dice_combo in combinations_with_replacement(dice_values, length):
                # Convert to list as that's what the method expects
                dice_list: List[int] = list(dice_combo)
                # Always get combinations for all 6 possible aside values
                result: List[CombinationResult] = StateTransition.getBestCombinationsForUpToNAside(dice_list, 6)
                # Store using tuple as key since lists aren't hashable
                lookup_table[dice_combo] = result
                
        return lookup_table
        
    @staticmethod
    def getBestCombinationsForUpToNAside(sorted_dice: List[int], n_aside: int) -> List[Optional[Tuple[List[int], int]]]:
        """
        Returns list of best (chosen values, score) tuples for each number of dice to be put aside 1..n
        Args:
            sorted_dice: List of dice values sorted in ascending order
            n_aside: Maximum number of dice that can be put aside
        Returns:
            List of length n_aside where each element is either None (no valid combination)
            or a tuple (list of dice values, score) for the best combination of that size
        """
        SEQUENCES = {
            (1, 2, 3, 4, 5): 500,
            (2, 3, 4, 5, 6): 750,
            (1, 2, 3, 4, 5, 6): 1500
        }
        
        def get_n_of_a_kind_score(value: int, count: int) -> int:
            if count < 3:
                return 0
            base_score = 1000 if value == 1 else value * 100
            return base_score * (2 ** (count - 3))
        
        def is_valid_single_die(value: int) -> bool:
            return value in [1, 5]
        
        def get_single_die_score(value: int) -> int:
            return 100 if value == 1 else 50 if value == 5 else 0
        
        def is_valid_combination(combo: List[int]) -> bool:
            """Check if this is a valid scoring combination (not just containing scoring dice)"""
            combo_tuple = tuple(sorted(combo))
            
            # Check if it's a sequence
            if combo_tuple in SEQUENCES:
                return True
                
            counts = Counter(combo)
            
            # For non-sequence combinations:
            remaining_dice = []
            for value, count in counts.items():
                if count >= 3:  # Valid n-of-a-kind
                    continue
                elif count < 3:  # Must be valid single scoring dice
                    if not is_valid_single_die(value):
                        return False
                    remaining_dice.extend([value] * count)
                    
            # If we have any remaining dice, they must all be scoring singles
            return all(is_valid_single_die(d) for d in remaining_dice)
        
        def score_combination(combo: List[int]) -> int:
            if not is_valid_combination(combo):
                return 0
                
            combo_tuple = tuple(sorted(combo))
            if combo_tuple in SEQUENCES:
                return SEQUENCES[combo_tuple]
            
            counts = Counter(combo)
            score = 0
            
            # Score three or more of a kind
            remaining_counts = counts.copy()
            for value, count in counts.items():
                if count >= 3:
                    score += get_n_of_a_kind_score(value, count)
                    remaining_counts[value] = 0
            
            # Score remaining single dice
            score += sum(get_single_die_score(value) * count 
                        for value, count in remaining_counts.items())
            
            return score

        # Initialize results list with None values
        results = [None] * n_aside
        
        # Process each possible number of dice
        for n in range(1, min(n_aside + 1, len(sorted_dice) + 1)):
            # Get all combinations of exactly n dice and their scores
            all_combos = list(combinations(sorted_dice, n))
            scored_combos = [(list(sorted(combo)), score_combination(combo)) for combo in all_combos]
            
            # Filter out zero-scoring combinations
            valid_combos = [(combo, score) for combo, score in scored_combos if score > 0]
            
            if valid_combos:
                # Find the combination with the highest score
                best_combo = max(valid_combos, key=lambda x: x[1])
                results[n-1] = best_combo
        
        return results
    


if __name__ == "__main__":
    lookup: LookupTable = StateTransition.generate_dice_combinations_lookup()
    
    # Example lookups
    print("\nExample lookups:")
    test_cases: List[DiceValues] = [
        (1,),           # Single 1
        (1, 1),         # Two 1s
        (1, 2, 3, 4, 5, 6),  # Full sequence
        (1, 1, 1, 5, 5, 5),
    ]

    for dice in test_cases:
        print(f"\nDice: {dice}")
        print(f"Results: {lookup[dice]}")