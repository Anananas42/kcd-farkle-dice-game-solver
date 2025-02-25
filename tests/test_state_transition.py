import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_transition import StateTransition

def run_test(description, dice, n_aside, expected):
    result = StateTransition.getBestCombinationsForUpToNAside(sorted(dice), n_aside)
    try:
        assert result == expected, f"\nExpected: {expected}\nGot: {result}"
        print(f"✓ {description}")
    except AssertionError as e:
        print(f"✗ {description}")
        print(e)

if __name__ == '__main__':
    print("\nBasic scoring combinations:")
    run_test("Single 1", [1], 1, [([1], 100)])
    run_test("Single 5", [5], 1, [([5], 50)])
    run_test("Non-scoring die", [2], 1, [None])
    run_test("Simple three of a kind", [1, 1, 1], 3, 
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000)])

    print("\nMultiple scoring dice:")
    # Bug potential: Handling multiple valid single-scoring dice
    run_test("Two ones", [1, 1], 2, 
            [([1], 100), ([1, 1], 200)])
    run_test("One and five", [1, 5], 2, 
            [([1], 100), ([1, 5], 150)])
    run_test("Mixed scoring and non-scoring", [1, 2, 5], 2, 
            [([1], 100), ([1, 5], 150)])

    print("\nSequences:")
    # Bug potential: Sequence detection and scoring
    run_test("Basic sequence 1-5", [1, 2, 3, 4, 5], 5, 
            [([1], 100), ([1, 5], 150), None, None, ([1, 2, 3, 4, 5], 500)])
    run_test("Sequence 2-6", [2, 3, 4, 5, 6], 5,
            [([5], 50), None, None, None, ([2, 3, 4, 5, 6], 750)])
    run_test("Full sequence", [1, 2, 3, 4, 5, 6], 6,
            [([1], 100), ([1, 5], 150), None, None, ([2, 3, 4, 5, 6], 750), ([1, 2, 3, 4, 5, 6], 1500)])

    print("\nN of a kind:")
    # Bug potential: Multiplier calculation
    run_test("Four of a kind", [1, 1, 1, 1], 4,
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000), ([1, 1, 1, 1], 2000)])
    run_test("Five of a kind", [1, 1, 1, 1, 1], 5,
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000), ([1, 1, 1, 1], 2000), ([1, 1, 1, 1, 1], 4000)])
    run_test("Six of a kind", [1, 1, 1, 1, 1, 1], 6,
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000), ([1, 1, 1, 1], 2000), 
            ([1, 1, 1, 1, 1], 4000), ([1, 1, 1, 1, 1, 1], 8000)])

    print("\nComplex combinations:")
    # Bug potential: Handling mixed scoring types
    run_test("Three of a kind plus singles", [1, 1, 1, 5, 5], 5,
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000), ([1, 1, 1, 5], 1050), ([1, 1, 1, 5, 5], 1100)])
    run_test("Two three of a kind", [1, 1, 1, 5, 5, 5], 6,
            [([1], 100), ([1, 1], 200), ([1, 1, 1], 1000), ([1, 1, 1, 5], 1050), 
            ([1, 1, 1, 5, 5], 1100), ([1, 1, 1, 5, 5, 5], 1500)])

    print("\nEdge cases:")
    # Bug potential: Boundary conditions
    run_test("Empty dice list", [], 1, [None])
    run_test("n_aside larger than dice count", [1], 3, [([1], 100), None, None])
    run_test("All non-scoring dice", [2, 3, 4, 6], 4, [None, None, None, None])
    run_test("All non-scoring dice", [2, 3, 4, 4, 6, 6], 6, [None, None, None, None, None, None])
    run_test("Mixed valid/invalid groups", [2, 2, 2, 1], 4,
            [([1], 100), None, ([2, 2, 2], 200), ([1, 2, 2, 2], 300)])