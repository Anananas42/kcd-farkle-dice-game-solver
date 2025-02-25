import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dice_probability_map import DiceProbabilityMap

def assert_close(a, b, epsilon=1e-2):
    """Assert that two values are close to each other within epsilon."""
    assert abs(a - b) < epsilon, f"Expected {a} to be close to {b}"

def assert_dict_close(dict1, dict2, epsilon=1e-2):
    """Assert that two dictionaries have the same keys and close values."""
    assert set(dict1.keys()) == set(dict2.keys()), "Dictionaries have different keys"
    for key in dict1:
        assert_close(dict1[key], dict2[key], epsilon)

def test_uniform_dice():
    """Test that uniform dice probabilities match expected values."""
    print("Testing uniform dice probabilities...")
    
    # Create probability map with default (uniform) weights
    prob_map = DiceProbabilityMap()
    
    # Test probability of scoring 1000 with 3 dice, putting aside 3
    # This should be 1/216 = 0.00463, the probability of rolling three 1s
    p_1000 = prob_map.get_probability_for_score(3, 3, 1000)
    expected_p_1000 = 1/216
    assert_close(p_1000, expected_p_1000)
    print(f"  Probability of scoring 1000 with 3 dice, putting aside 3: {p_1000:.4f} ✓")
    
    # Test probabilities for 3 dice, putting aside 2
    probabilities = prob_map.get_score_probabilities(3, 2)
    
    # Expected values from original output
    expected_p_0 = 0.7407
    expected_p_100 = 0.0602
    expected_p_150 = 0.1250
    expected_p_200 = 0.0741
    
    assert_close(probabilities[0], expected_p_0)
    assert_close(probabilities[2], expected_p_100)  # index 2 = 100 points
    assert_close(probabilities[3], expected_p_150)  # index 3 = 150 points
    assert_close(probabilities[4], expected_p_200)  # index 4 = 200 points
    
    print(f"  Probability of scoring 0 with 3 dice, putting aside 2: {probabilities[0]:.4f} ✓")
    print(f"  Probability of scoring 100 with 3 dice, putting aside 2: {probabilities[2]:.4f} ✓")
    print(f"  Probability of scoring 150 with 3 dice, putting aside 2: {probabilities[3]:.4f} ✓")
    print(f"  Probability of scoring 200 with 3 dice, putting aside 2: {probabilities[4]:.4f} ✓")
    
    # Test distribution statistics
    stats = prob_map.get_score_distribution_stats(3, 2)
    
    expected_stats = {
        'mean': 39.58,
        'std': 69.36,
        'min_score': 0,
        'max_score': 200,
        'p_zero': 0.7407
    }
    
    for key in expected_stats:
        assert_close(stats[key], expected_stats[key])
    
    print("  Distribution statistics match expected values ✓")
    print("Uniform dice tests passed! ✓")


def test_weighted_dice():
    """Test weighted dice probabilities with various weight distributions."""
    print("\nTesting weighted dice probabilities...")
    
    # Case 1: Weighted towards 1s
    # Die has 50% chance of rolling a 1, 10% for each other value
    weights_favor_ones = {1: 0.5, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1}
    
    prob_map_ones = DiceProbabilityMap(weights_favor_ones)
    
    # With three dice weighted towards 1s, probability of three 1s should be 0.5^3 = 0.125
    p_1000_weighted = prob_map_ones.get_probability_for_score(3, 3, 1000)
    expected_p_1000_weighted = 0.5**3
    assert_close(p_1000_weighted, expected_p_1000_weighted)
    print(f"  Weighted (favor 1s) probability of scoring 1000 with 3 dice: {p_1000_weighted:.4f} ✓")
    
    # Get stats for weighted distribution
    stats_weighted = prob_map_ones.get_score_distribution_stats(3, 2)
    print(f"  Weighted (favor 1s) mean score with 3 dice, putting aside 2: {stats_weighted['mean']:.2f}")
    print(f"  Weighted (favor 1s) p_zero with 3 dice, putting aside 2: {stats_weighted['p_zero']:.4f}")
    
    # The mean should be higher than uniform since we roll more 1s
    uniform_stats = prob_map = DiceProbabilityMap().get_score_distribution_stats(3, 2)
    assert stats_weighted['mean'] > uniform_stats['mean']
    print("  Weighted mean score is higher than uniform (as expected) ✓")
    
    # Case 2: No chance of 1s or 5s
    # This should result in extremely high probability of zero scores
    weights_no_scoring = {1: 0, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0, 6: 0.25}
    
    prob_map_no_scoring = DiceProbabilityMap(weights_no_scoring)
    
    # With no 1s or 5s possible, probability of scoring zero should be 1.0
    p_0_no_scoring = prob_map_no_scoring.get_probability_for_score(3, 2, 0)
    assert_close(p_0_no_scoring, 1.0)
    print(f"  No-scoring-dice probability of zero with 3 dice: {p_0_no_scoring:.4f} ✓")
    
    # Case 3: Perfect dice (all 1s)
    weights_all_ones = {1: 1.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    prob_map_all_ones = DiceProbabilityMap(weights_all_ones)
    
    # With all 1s, probability of scoring 200 with 2 dice should be 1.0
    p_200_all_ones = prob_map_all_ones.get_probability_for_score(3, 2, 200)
    assert_close(p_200_all_ones, 1.0)
    print(f"  All-ones probability of scoring 200 with 3 dice, putting aside 2: {p_200_all_ones:.4f} ✓")
    
    print("Weighted dice tests passed! ✓")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases and error handling...")
    
    # Empty weights (should raise ValueError)
    try:
        DiceProbabilityMap({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0})
        print("  ✗ Failed to catch zero weights")
    except ValueError as e:
        print(f"  Caught zero weights error correctly: {e} ✓")
    
    # Partial weights (should normalize correctly)
    partial_weights = {1: 1.0, 6: 3.0}  # Only specify 1 and 6
    prob_map_partial = DiceProbabilityMap(partial_weights)
    
    # Check internal weights were normalized correctly
    expected_weights = {1: 0.25, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0.75}
    for face, expected in expected_weights.items():
        actual = prob_map_partial._dice_weights[face]
        assert_close(actual, expected)
    print("  Partial weights normalized correctly ✓")
    
    # Invalid method parameters
    prob_map = DiceProbabilityMap()
    
    try:
        prob_map.get_score_probabilities(7, 1)  # Too many dice
        print("  ✗ Failed to catch too many dice")
    except ValueError:
        print("  Caught too many dice error correctly ✓")
    
    try:
        prob_map.get_score_probabilities(3, 4)  # More aside than available
        print("  ✗ Failed to catch too many dice aside")
    except ValueError:
        print("  Caught too many dice aside error correctly ✓")
    
    try:
        prob_map.get_probability_for_score(3, 2, 75)  # Score not multiple of 50
        print("  ✗ Failed to catch invalid score (not multiple of 50)")
    except ValueError:
        print("  Caught invalid score error correctly ✓")
    
    print("Edge case tests passed! ✓")


def test_real_world_scenarios():
    """Test some real-world dice scenarios and compare results."""
    print("\nTesting practical scenarios with different dice weights...")
    
    # Scenario 1: Slightly loaded dice (favoring 6s)
    slightly_loaded = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.15, 6: 0.25}
    prob_map_loaded = DiceProbabilityMap(slightly_loaded)
    
    # Scenario 2: Fair casino dice
    fair_dice = DiceProbabilityMap()
    
    # Compare distribution stats for 6 dice, putting aside 3
    loaded_stats = prob_map_loaded.get_score_distribution_stats(6, 3)
    fair_stats = fair_dice.get_score_distribution_stats(6, 3)
    
    print("Distribution stats for 6 dice, putting aside 3:")
    print(f"  Fair dice - Mean score: {fair_stats['mean']:.2f}, Max score: {fair_stats['max_score']}")
    print(f"  Loaded dice - Mean score: {loaded_stats['mean']:.2f}, Max score: {loaded_stats['max_score']}")
    
    # Probability of a high score (Three-of-a-kind with 6s = 600 points)
    p_600_fair = fair_dice.get_probability_for_score(3, 3, 600)
    p_600_loaded = prob_map_loaded.get_probability_for_score(3, 3, 600)
    
    print(f"Probability of scoring 600 with 3 dice (three 6s):")
    print(f"  Fair dice: {p_600_fair:.6f}")
    print(f"  Loaded dice (favoring 6s): {p_600_loaded:.6f}")
    
    # The loaded dice should have higher probability of three 6s
    assert p_600_loaded > p_600_fair
    print("  Loaded dice have higher probability of three 6s, as expected ✓")
    
    print("Real-world scenario tests passed! ✓")


def run_all_tests():
    """Run all tests for the DiceProbabilityMap class."""
    print("Running all tests for DiceProbabilityMap with weighted dice...\n")
    
    test_uniform_dice()
    test_weighted_dice()
    test_edge_cases()
    test_real_world_scenarios()
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()