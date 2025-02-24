from itertools import combinations, combinations_with_replacement
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math

class StateTransition:
    def __init__(self, sorted_dice: List[int],  n_set_aside: int):
        self.n_set_aside = n_set_aside