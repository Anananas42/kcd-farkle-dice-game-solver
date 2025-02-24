from itertools import combinations, combinations_with_replacement
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math

class StateTransition:
    def __init__(self, sorted_dice: List[int],  n_set_aside: int):
        self.n_set_aside = n_set_aside
        self.value = self.getBestCombinationValue(sorted_dice, n_set_aside)

    def getBestCombinationValue(sorted_dice: List[int],  n_set_aside: int) -> int:
        return None
    
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
            result: List[CombinationResult] = getBestCombinationsForUpToNAside(dice_list, 6)
            # Store using tuple as key since lists aren't hashable
            lookup_table[dice_combo] = result
            
    return lookup_table
        
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

def get_permutation_count(dice_combo: Tuple[int, ...]) -> int:
    """
    Calculate the number of unique permutations for a combination of dice.
    Uses multinomial coefficient to handle repeated values correctly.
    
    Args:
        dice_combo: Tuple of dice values (sorted)
    
    Returns:
        Number of unique permutations
    """
    n = len(dice_combo)
    # Count occurrences of each value
    value_counts = list(Counter(dice_combo).values())
    
    # Calculate multinomial coefficient: n! / (n1! * n2! * ... * nk!)
    # where n is total length and ni are counts of each unique value
    result = math.factorial(n)
    for count in value_counts:
        result //= math.factorial(count)
    
    return result

def aggregate_scores_for_n_dice(n: int, lookup_table: Dict[Tuple[int, ...], List[Any]]) -> List[np.ndarray]:
    """
    Aggregates all possible scores for n dice when putting aside m dice (for m in 1..n).
    None results are counted as score 0. Takes into account all possible permutations
    of each dice combination.
    
    Args:
        n: Number of dice rolled
        lookup_table: Precomputed lookup table from generate_dice_combinations_lookup()
    
    Returns:
        List[np.ndarray]: List of length n, where each element is a numpy array representing
        the frequency of scores for putting aside m dice (m is the index + 1).
        Each array index i represents the frequency of score i*50.
    """
    max_score = 8000  # Maximum possible score with 6 dice
    array_size = max_score // 50 + 1  # Size needed to represent all possible scores
    
    # Initialize list of arrays for each m in 1..n
    score_frequencies = [np.zeros(array_size, dtype=int) for _ in range(n)]
    
    # Generate all possible n-dice combinations
    for dice_combo in combinations_with_replacement(range(1, 7), n):
        # Calculate number of permutations for this combination
        perm_count = get_permutation_count(dice_combo)
        
        # Get the scoring combinations for this dice set
        combinations = lookup_table[dice_combo]
        
        # For each number of dice we might put aside (m from 1 to n)
        for m in range(n):
            combination = combinations[m]  # Get the best combination for putting aside m+1 dice
            if combination is not None:
                # Extract score and convert to array index
                _, score = combination
                score_index = score // 50
                # Increment the frequency by the number of permutations
                score_frequencies[m][score_index] += perm_count
            else:
                # Count None as score 0
                score_frequencies[m][0] += perm_count
    
    return score_frequencies

def print_score_frequencies(score_frequencies: List[np.ndarray]):
    """
    Helper function to print score frequencies in a readable format.
    Includes score 0 frequencies from None results.
    """
    for m, frequencies in enumerate(score_frequencies, 1):
        print(f"\nPutting aside {m} dice:")
        nonzero_indices = np.nonzero(frequencies)[0]
        if len(nonzero_indices) == 0:
            print("  No valid scores")
            continue
        
        # Always print score 0 frequency (includes None results)
        print(f"  Score 0: {frequencies[0]} occurrences")
        
        # Print other scores
        for idx in nonzero_indices:
            if idx == 0:  # Skip 0 as we've already printed it
                continue
            score = idx * 50
            count = frequencies[idx]
            print(f"  Score {score}: {count} occurrences")

def generate_all_frequency_distributions(lookup_table: Dict[Tuple[int, ...], List[Any]]) -> Dict[int, List[np.ndarray]]:
    """
    Generates frequency distributions for all possible dice counts (1-6) and all possible
    numbers of dice that could be put aside.
    
    Args:
        lookup_table: Precomputed lookup table from generate_dice_combinations_lookup()
    
    Returns:
        Dict[int, List[np.ndarray]]: A dictionary where:
            - key: number of dice rolled (1-6)
            - value: list of numpy arrays, where index i contains frequencies for putting 
                    aside i+1 dice. Each array contains frequencies for scores 0, 50, 100, etc.
    """
    distributions: Dict[int, List[np.ndarray]] = {}
    
    # Generate distributions for each possible number of dice (1-6)
    for n_dice in range(1, 7):
        distributions[n_dice] = aggregate_scores_for_n_dice(n_dice, lookup_table)
    
    return distributions

def print_all_distributions(distributions: Dict[int, List[np.ndarray]]):
    """
    Prints all frequency distributions in a readable format.
    """
    for n_dice in sorted(distributions.keys()):
        print(f"\n=== Rolling {n_dice} dice ===")
        print_score_frequencies(distributions[n_dice])

def calculate_probability_distribution(frequencies: np.ndarray) -> np.ndarray:
    """
    Converts frequency counts to probability distribution.
    
    Args:
        frequencies: Array of frequency counts
    
    Returns:
        Array of probabilities (sums to 1)
    """
    total = np.sum(frequencies)
    return frequencies / total if total > 0 else frequencies
        
def create_probability_matrix(lookup_table: Dict[Tuple[int, ...], List[Any]]) -> np.ndarray:
    """
    Creates a 3D numpy array for efficient probability lookups.
    
    Args:
        lookup_table: Precomputed lookup table from generate_dice_combinations_lookup()
    
    Returns:
        np.ndarray: A 3D array where:
            - First dimension (n): number of dice available (1-6)
            - Second dimension (m): number of dice to set aside (1-6)
            - Third dimension: probabilities for scores [0, 50, 100, ..., 8000]
            
        Access pattern: probabilities[n-1][m-1] gives probability array for
        n dice available and m dice set aside.
        Note: m must be <= n, invalid combinations will contain zeros
    """
    max_score = 8000
    score_array_size = max_score // 50 + 1
    
    # Initialize 3D array: [6 dice counts][6 possible aside][score probabilities]
    probabilities = np.zeros((6, 6, score_array_size), dtype=float)
    
    # Generate all distributions
    distributions = generate_all_frequency_distributions(lookup_table)
    
    # Fill the array with normalized probabilities
    for n_dice in range(1, 7):  # 1-6 dice
        freq_list = distributions[n_dice]
        for m_aside in range(len(freq_list)):  # 0 to n_dice-1 indexes
            frequencies = freq_list[m_aside]
            total = np.sum(frequencies)
            if total > 0:  # Avoid division by zero
                probabilities[n_dice-1][m_aside] = frequencies / total
    
    return probabilities

if __name__ == "__main__":
    lookup_table: LookupTable = generate_dice_combinations_lookup()
    # distributions = generate_all_frequency_distributions(lookup_table)
    # # Example lookups
    # print("\nExample lookups:")
    # test_cases: List[DiceValues] = [
    #     (1,),           # Single 1
    #     (1, 1),         # Two 1s
    #     (1, 2, 3, 4, 5, 6),  # Full sequence
    #     (1, 1, 1, 5, 5, 5),
    # ]

    # for dice in test_cases:
    #     print(f"\nDice: {dice}")
    #     print(f"Results: {lookup_table[dice]}")

    # # Example with 2 dice
    # print("\nAnalyzing scores for 2 dice:")
    # frequencies_2_dice = aggregate_scores_for_n_dice(2, lookup_table)
    # print_score_frequencies(frequencies_2_dice)
    
    # # Example with 3 dice
    # print("\nAnalyzing scores for 3 dice:")
    # frequencies_3_dice = aggregate_scores_for_n_dice(3, lookup_table)
    # print_score_frequencies(frequencies_3_dice)

    # print("\nComplete frequency distributions:")
    # # print_all_distributions(distributions)
    
    # # Example: Calculate probabilities for specific case
    # n_dice = 6
    # m_aside = 1  # putting aside 2 dice
    # frequencies = distributions[n_dice][m_aside - 1]
    # probabilities = calculate_probability_distribution(frequencies)
    
    # print(f"\nProbabilities for {n_dice} dice, putting aside {m_aside} dice:")
    # for score, prob in enumerate(probabilities):
    #     if prob > 0:
    #         print(f"  Score {score * 50}: {prob:.3f}")

    print(create_probability_matrix(lookup_table).shape)