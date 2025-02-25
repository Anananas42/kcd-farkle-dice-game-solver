from itertools import combinations, combinations_with_replacement
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import math

DiceValues = Tuple[int, ...]  # Tuple of 1-6 integers
ScoringCombination = Tuple[List[int], int]  # (dice_list, score)
CombinationResult = Optional[ScoringCombination]  # None or (dice_list, score)
LookupTable = Dict[DiceValues, List[CombinationResult]]

class DiceProbabilityMap:
    def __init__(self, dice_weights: Optional[Dict[int, float]] = None):
        """
        Initializes the probability map with optional dice weights.
        
        Args:
            dice_weights: Optional dictionary mapping die values (1-6) to weights.
                        If None, uniform probability (equal weights) is assumed.
        """
        # Normalize weights if provided, otherwise use uniform weights
        self._dice_weights = self.__normalize_weights(dice_weights)
        self._lookup_table = self.__generate_dice_combinations_lookup()
        self._probability_matrix = self.__create_probability_matrix(self._lookup_table)

    def __normalize_weights(self, weights: Optional[Dict[int, float]]) -> Dict[int, float]:
        """
        Normalizes the provided weights to ensure they sum to 1.
        If weights are None, returns uniform probability distribution.
        
        Args:
            weights: Dictionary mapping die values (1-6) to weights
            
        Returns:
            Dictionary with normalized weights that sum to 1
        """
        if weights is None:
            # Default to uniform distribution
            return {i: 1/6 for i in range(1, 7)}
        
        # Ensure all faces have weights
        normalized = {i: weights.get(i, 0.0) for i in range(1, 7)}
        
        # Normalize to sum to 1
        weight_sum = sum(normalized.values())
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        return {face: weight/weight_sum for face, weight in normalized.items()}

    def __generate_dice_combinations_lookup(self) -> LookupTable:
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
                result: List[CombinationResult] = self.__getBestCombinationsForUpToNAside(dice_list, 6)
                # Store using tuple as key since lists aren't hashable
                lookup_table[dice_combo] = result
                
        return lookup_table
    
    def __getBestCombinationsForUpToNAside(self, sorted_dice: List[int], n_aside: int) -> List[Optional[Tuple[List[int], int]]]:
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
    

    def __create_probability_matrix(self, lookup_table: Dict[Tuple[int, ...], List[Any]]) -> np.ndarray:
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
        distributions = self.__generate_all_frequency_distributions(lookup_table)
        
        # Fill the array with normalized probabilities
        for n_dice in range(1, 7):  # 1-6 dice
            freq_list = distributions[n_dice]
            for m_aside in range(len(freq_list)):  # 0 to n_dice-1 indexes
                frequencies = freq_list[m_aside]
                total = np.sum(frequencies)
                if total > 0:  # Avoid division by zero
                    probabilities[n_dice-1][m_aside] = frequencies / total
        
        return probabilities
    
    def get_score_probabilities(self, n_dice: int, m_aside: int) -> np.ndarray:
        """
        Get probability distribution for scores when putting aside m dice out of n available.
        
        Args:
            n_dice: Number of dice available (1-6)
            m_aside: Number of dice to put aside (must be <= n_dice)
            
        Returns:
            np.ndarray: Array of probabilities for scores [0, 50, 100, ..., 8000]
            where index i corresponds to score i*50
            
        Raises:
            ValueError: If n_dice or m_aside are invalid
        """
        if not 1 <= n_dice <= 6:
            raise ValueError("Number of dice must be between 1 and 6")
        if not 1 <= m_aside <= n_dice:
            raise ValueError("Number of dice to put aside must be between 1 and number of dice available")
            
        return self._probability_matrix[n_dice-1][m_aside-1]
    
    def get_probability_for_score(self, n_dice: int, m_aside: int, score: int) -> float:
        """
        Get probability of achieving a specific score when putting aside m dice out of n available.
        
        Args:
            n_dice: Number of dice available (1-6)
            m_aside: Number of dice to put aside (must be <= n_dice)
            score: Target score (must be multiple of 50)
            
        Returns:
            float: Probability of achieving the score
            
        Raises:
            ValueError: If parameters are invalid
        """
        if score % 50 != 0:
            raise ValueError("Score must be a multiple of 50")
        if score < 0 or score > 8000:
            raise ValueError("Score must be between 0 and 8000")
            
        probabilities = self.get_score_probabilities(n_dice, m_aside)
        return probabilities[score // 50]
    
    def get_best_score_probability(self, n_dice: int, m_aside: int) -> Tuple[int, float]:
        """
        Get the highest possible score and its probability for given dice configuration.
        
        Args:
            n_dice: Number of dice available (1-6)
            m_aside: Number of dice to put aside (must be <= n_dice)
            
        Returns:
            Tuple[int, float]: (best possible score, probability of achieving it)
        """
        probabilities = self.get_score_probabilities(n_dice, m_aside)
        nonzero_indices = np.nonzero(probabilities)[0]
        if len(nonzero_indices) == 0:
            return (0, 1.0)  # All probability mass at zero
            
        best_score_idx = nonzero_indices[-1]
        return (best_score_idx * 50, probabilities[best_score_idx])
    
    def get_score_distribution_stats(self, n_dice: int, m_aside: int) -> Dict[str, Union[float, int]]:
        """
        Get statistical information about the score distribution.
        
        Args:
            n_dice: Number of dice available (1-6)
            m_aside: Number of dice to put aside (must be <= n_dice)
            
        Returns:
            Dict with keys:
                - mean: Expected score
                - std: Standard deviation of scores
                - min_score: Minimum possible score
                - max_score: Maximum possible score
                - p_zero: Probability of scoring zero
        """
        probabilities = self.get_score_probabilities(n_dice, m_aside)
        scores = np.arange(len(probabilities)) * 50
        
        return {
            'mean': np.sum(scores * probabilities),
            'std': np.sqrt(np.sum((scores - np.sum(scores * probabilities))**2 * probabilities)),
            'min_score': scores[np.nonzero(probabilities)[0][0]] if np.any(probabilities > 0) else 0,
            'max_score': scores[np.nonzero(probabilities)[0][-1]] if np.any(probabilities > 0) else 0,
            'p_zero': probabilities[0]
        }
    
    def __generate_all_frequency_distributions(self, lookup_table: Dict[Tuple[int, ...], List[Any]]) -> Dict[int, List[np.ndarray]]:
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
            distributions[n_dice] = self.__aggregate_scores_for_n_dice(n_dice, lookup_table)
        
        return distributions
    
    def __aggregate_scores_for_n_dice(self, n: int, lookup_table: Dict[Tuple[int, ...], List[Any]]) -> List[np.ndarray]:
        max_score = 8000
        array_size = max_score // 50 + 1
        
        score_frequencies = [np.zeros(array_size, dtype=float) for _ in range(n)]
        
        # Generate all possible n-dice combinations
        for dice_combo in combinations_with_replacement(range(1, 7), n):
            # Calculate weighted probability for this combination
            combo_probability = self.__get_combination_probability(dice_combo)
            
            # Get the scoring combinations for this dice set
            combinations = lookup_table[dice_combo]
            
            # For each number of dice we might put aside (m from 1 to n)
            for m in range(n):
                combination = combinations[m]
                if combination is not None:
                    _, score = combination
                    score_index = score // 50
                    # Add weighted probability instead of count
                    score_frequencies[m][score_index] += combo_probability
                else:
                    score_frequencies[m][0] += combo_probability
        
        return score_frequencies
    
    def __get_combination_probability(self, dice_combo: Tuple[int, ...]) -> float:
        """
        Calculate the probability of getting a specific dice combination
        with the current die weights.
        
        Args:
            dice_combo: Tuple of dice values
            
        Returns:
            Probability of getting this combination
        """
        # Count occurrences of each value
        counter = Counter(dice_combo)
        
        # Calculate multinomial probability
        n = len(dice_combo)
        
        # Multinomial coefficient
        coefficient = math.factorial(n)
        for count in counter.values():
            coefficient //= math.factorial(count)
        
        # Multiply by probabilities for each face value
        probability = coefficient
        for face, count in counter.items():
            probability *= (self._dice_weights[face] ** count)
        
        return probability

if __name__ == "__main__":
    prob_map = DiceProbabilityMap()
    
    # Example: Get probability distribution for 3 dice, putting aside 2
    probs = prob_map.get_score_probabilities(3, 2)
    print("\nProbabilities for 3 dice, putting aside 2:")
    for score, prob in enumerate(probs):
        if prob > 0:
            print(f"Score {score * 50}: {prob:.4f}")
    
    # Example: Get probability of specific score
    print(f"\nProbability of scoring 1000 with 3 dice, putting aside 3: "
          f"{prob_map.get_probability_for_score(3, 3, 1000):.4f}")
    
    # Example: Get best possible score and probability
    score, prob = prob_map.get_best_score_probability(3, 3)
    print(f"\nBest possible score with 3 dice, putting aside 3: {score} (probability: {prob:.4f})")
    
    # Example: Get distribution statistics
    stats = prob_map.get_score_distribution_stats(3, 2)
    print("\nDistribution statistics for 3 dice, putting aside 2:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")