"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
from .utils import check_random_state

data_size = 0
from . import genetic
from .treelstm_const_attention import AttentiveMixedArityTreeLSTM

class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None, 
                 y_predicts=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.y_predicts = y_predicts

        dim_total = int(len(function_set) + n_features)
        dim_total += 1 # deal with constant

        embedding_dim = 8
        hidden_dim = genetic.treelstm_dim
        operator_encode = genetic.operator_enc
        subtree_encode = genetic.subtree_enc

        self.model = AttentiveMixedArityTreeLSTM(vocab_size=dim_total, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def mystr(self, program):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        # program = self._program
        for i, node in enumerate(program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    output += 'X%s' % node
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(program) - 1:
                    output += ', '
        return output

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def update_y_predicts(self, X):
        """Evaluate program and store predicted y-values at each node."""

        y_dict = {}  # index -> value
        eval_stack = []

        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                eval_stack.append((i, node, []))  # (index, function, list of args)
            else:
                # variable or constant
                val = X[:, node] if isinstance(node, int) else np.full(X.shape[0], node)
                y_dict[i] = val  # store result at index i

                # append to the latest function waiting for operands
                if eval_stack:
                    eval_stack[-1][2].append(val)

                # collapse function nodes if their operands are ready
                while eval_stack and len(eval_stack[-1][2]) == eval_stack[-1][1].arity:
                    idx_f, func, args = eval_stack.pop()
                    result = func(*args)
                    y_dict[idx_f] = result
                    if eval_stack:
                        eval_stack[-1][2].append(result)

        # Rebuild y_predicts in program order
        y_predicts = [y_dict[i] for i in range(len(self.program))]

        self.y_predicts = y_predicts

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        global op_count
        op_count = 0
        global data_size
        data_size = X.shape[0]
        # print(f'data_size: {data_size}')
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                op_count += 1*data_size
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        self.update_y_predicts(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end
    
    def get_subtree_depth_limit(self, random_state, program=None, max_depth=3, max_retry=50):

        if program is None:
            program = self.program

        for _ in range(max_retry):

            probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
            probs = np.cumsum(probs / probs.sum())
            start = np.searchsorted(probs, random_state.uniform())

            stack = 1
            end = start

            while stack > end - start:
                node = program[end]
                if isinstance(node, _Function):
                    stack += node.arity
                end += 1
            subtree = program[start:end]
            depth = self.getDepth(subtree)

            if depth <= max_depth:
                return start, end

        return start, end
    
    def getDepth(self, program): # start from 0
        terminals = [0]
        depth = 1

        for node in program:

            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)

            else:
                terminals[-1] -= 1

                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1

        return depth - 1 

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def getUpdateOpevals(self, program, start, total_depth):
        """
        Calculate the number of nodes that need to be updated after crossover.
        
        :param program: List of nodes representing the program tree
        :param start: Start index of the subtree being replaced
        :param total_depth: Total depth of the tree
        :return: Number of nodes that need to be updated
        """
        # Initialize depth map
        depth_map = [None] * len(program)
        
        def calculate_depth(index, current_depth):
            """
            Recursively calculate depth for each node.
            :param index: Current node index
            :param current_depth: Depth of the current node
            :return: Next index to continue
            """
            depth_map[index] = current_depth
            node = program[index]
            
            # If it's a function, recursively calculate its children
            if isinstance(node, _Function):
                next_index = index + 1
                for _ in range(node.arity):
                    next_index = calculate_depth(next_index, current_depth + 1)
                return next_index
            
            # If it's a terminal, just go to the next node
            return index + 1

        # Start from root
        calculate_depth(0, 0)

        # Get the depth of the subtree to be cut
        cut_depth = depth_map[start]
        count = cut_depth - depth_map[0]
        assert(count >= 0)

        return count, depth_map
    
    def getAnotherSubtree(self, start, end):
        # Get the receiver (the entire program) and the left subtree based on the given start and end indices.
        receiver = self.program
        left = receiver[start:end]

        # If the left subtree is the entire receiver (no distinct subtree), return None.
        if left == receiver:
            return start, start, end

        from .functions import _function_map
        # Define one-arity functions that only expect one child (e.g., sin, cos, exp).
        one_arity_funcs = {name for name, func in _function_map.items() if func.arity == 1}

        # Define two-arity functions that expect two children (e.g., add, sub, mul, div, pow).
        two_arity_funcs = {name for name, func in _function_map.items() if func.arity == 2}

        # Find the parent node of the given subtree.
        def find_parent(prog, start):
            """Return the parent node's index and the number of children it expects."""
            stack = []
            for i in range(start):
                if isinstance(prog[i], _Function):
                    token = prog[i].name
                else:
                    token = prog[i]
                # print(f'token: {token}')
                if token in two_arity_funcs:
                    stack.append((i, 2))  # Expecting two children for two-arity functions
                elif token in one_arity_funcs:
                    stack.append((i, 1))  # Expecting one child for one-arity functions
                else:
                    # For terminal nodes (variables or constants), reduce the expected children count
                    while stack:
                        idx, needed = stack.pop()
                        needed -= 1
                        if needed > 0:
                            stack.append((idx, needed))
                            break
                        else:
                            continue
            # Return the parent node index which contains the subtree starting from 'start'
            return stack[-1][0] if stack else None

        parent_idx = find_parent(receiver, start)
        if parent_idx is None:
            return start, start, end  # No parent found, could indicate an error or root node

        # Get the parent operation (either one-arity or two-arity function)
        parent_op = receiver[parent_idx].name

        # If the parent is a one-arity function, return the left subtree (no other side)
        if parent_op in one_arity_funcs:
            return parent_idx, start, end  # Only left subtree exists for one-arity functions
        elif parent_op in two_arity_funcs:
            # Find the start indices for both children of the two-arity parent
            def get_children_starts(prog, parent_idx):
                child1 = parent_idx + 1
                stack = 1
                i = child1
                while stack > 0:
                    node = self.program[i]
                    if isinstance(node, _Function):
                        stack += node.arity
                    stack -= 1
                    i += 1
                child2 = i
                return child1, child2

            left_idx, right_idx = get_children_starts(receiver, parent_idx)
            stack = 1
            end_r = right_idx
            while stack > end_r - right_idx:
                node = self.program[end_r]
                if isinstance(node, _Function):
                    stack += node.arity
                end_r += 1
            return parent_idx, right_idx, end_r

    def buildMyTree(self, tokens, subtree_encode):
        """
        Convert gplearn-style prefix tokens to TreeNode structure.
        Example: [add_func, 0] => TreeNode('add', TreeNode(0), TreeNode(0))
        """
        if subtree_encode == 0:
            from .code2vec import TreeNode
        elif subtree_encode == 1:
            from .treelstm_const import TreeNode
        elif subtree_encode == 2:
            from .treelstm_const_attention import TreeNode

        operator_to_id = {func.name: i for i, func in enumerate(self.function_set)}

        def _build(tokens, idx):
            token = tokens[idx]
            idx += 1

            if hasattr(token, 'arity'):  # gplearn _Function object
                arity = token.arity
                name = token.name
                token_id = operator_to_id.get(name, -1)
                if arity == 1:
                    left, idx = _build(tokens, idx)
                    return TreeNode(token_id, left=left), idx
                elif arity == 2:
                    left, idx = _build(tokens, idx)
                    right, idx = _build(tokens, idx)
                    return TreeNode(token_id, left=left, right=right), idx
            else:
                return TreeNode(token), idx

        root, _ = _build(tokens, 0)
        return root

    def getOperatorEmbed(self, parent_idx, operator_enc_mode=1):
        # Define the base mapping (matching function_set)
        operator_to_id = {func.name: i for i, func in enumerate(self.function_set)}
        num_functions = len(self.function_set)
        node = self.program[parent_idx]

        if isinstance(node, _Function):
            parent_node = node.name
            parent_embed = operator_to_id.get(parent_node, num_functions)
        else:
            parent_embed = parent_idx + num_functions
        
        return parent_embed

    def crossover(self, donor, random_state, op_limit): # subtree crossover
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))

        global all_opevals
        from .genetic import all_opevals
        update_op, _ = self.getUpdateOpevals(self.program, start, self._depth())
        crossover_op_cost = data_size * update_op
        if update_op != 0: assert(crossover_op_cost != 0)
        if all_opevals + crossover_op_cost >= op_limit:
            genetic.stop = True
            crossover_op_cost = 0

        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed, crossover_op_cost

    def OSDXO(self, dim, mlp_predict, donor, donor_y_predicts, random_state, op_limit):
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))

        parent_idx, start_r, end_r = self.getAnotherSubtree(start, end)
        parent_embed = self.getOperatorEmbed(parent_idx, genetic.operator_enc)

        yr_tree = self.buildMyTree(self.program[start_r:end_r], genetic.subtree_enc) # another subtree
        ys_tree = self.buildMyTree(self.program[start:end], genetic.subtree_enc) # selected receiver
        yd_tree = self.buildMyTree(donor[donor_start:donor_end], genetic.subtree_enc) # donor subtree

        yr_embed, _ = self.model(yr_tree)
        ys_embed, _ = self.model(ys_tree)
        yd_embed, _ = self.model(yd_tree)
        
        predict_data = [parent_embed]
        predict_data.extend(yr_embed.detach().numpy()[0].tolist())
        operator_data = [parent_embed]

        subtree_data = []
        subtree_data.extend(yr_embed.detach().numpy()[0].tolist())
        receiver_data = []
        receiver_data.extend(ys_embed.detach().numpy()[0].tolist())
        donor_data = []
        donor_data.extend(yd_embed.detach().numpy()[0].tolist())

        global all_opevals
        from .genetic import all_opevals

        if mlp_predict:
            min_dist = float('inf')
            best_cand = None
            best_cand_start = 0
            best_cand_end = 0
            best_yc_embed = []
            input_data = np.array(predict_data)
            results = genetic.mlp_relu_decay_model.predict(input_data)
            candidates, candidates_y = genetic.getCandidates(2, 10, random_state)
            
            for i in range(len(candidates)):
                cand_prog = candidates[i].program
                cand_start, cand_end = self.get_subtree(random_state, cand_prog)
                if best_cand == None:
                    best_cand = cand_prog
                    best_cand_start = cand_start
                    best_cand_end = cand_end
                yc = candidates_y[i][cand_start]
                yc_tree = self.buildMyTree(cand_prog[cand_start:cand_end], genetic.subtree_enc)
                yc_embed, _ = self.model(yc_tree)
                # Euclidean distance
                dist = np.sqrt(np.sum((results - yc_embed.detach().numpy()) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand_prog
                    best_cand_start = cand_start
                    best_cand_end = cand_end
                    best_yc_embed = yc_embed

            donor_data = []
            donor_data.extend(best_yc_embed.detach().numpy()[0].tolist())
                
            update_op, _ = self.getUpdateOpevals(self.program, start, self._depth())
            crossover_op_cost = data_size * update_op
            if update_op != 0: assert(crossover_op_cost != 0)
            if len(self.program[:start] +
                    best_cand[best_cand_start:best_cand_end] +
                    self.program[end:]) >= genetic.max_length or all_opevals + crossover_op_cost >= op_limit:
                crossover_op_cost = 0
                genetic.stop = True
                return self.program, removed, donor_removed, crossover_op_cost, operator_data, subtree_data, receiver_data, donor_data
            else:
                return (self.program[:start] +
                    best_cand[best_cand_start:best_cand_end] +
                    self.program[end:]), removed, donor_removed, crossover_op_cost, operator_data, subtree_data, receiver_data, donor_data

        update_op, _ = self.getUpdateOpevals(self.program, start, self._depth())
        crossover_op_cost = data_size * update_op
        if update_op != 0: assert(crossover_op_cost != 0)
        if all_opevals + crossover_op_cost >= op_limit:
            crossover_op_cost = 0

        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed, crossover_op_cost, operator_data, subtree_data, receiver_data, donor_data

    def subtree_mutation(self, random_state, op_limit):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state, op_limit)

    def hoist_mutation(self, random_state, op_limit):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))

        global all_opevals
        from .genetic import all_opevals
        update_op, _ = self.getUpdateOpevals(self.program, start, self._depth())
        hoist_mut_op_cost = data_size * update_op

        if all_opevals + hoist_mut_op_cost >= op_limit:
            hoist_mut_op_cost = 0
            genetic.stop = True
            return self.program, removed, hoist_mut_op_cost
        else:
            return self.program[:start] + hoist + self.program[end:], removed, hoist_mut_op_cost

        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state, op_limit):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        # from . import genetic
        global all_opevals
        from .genetic import all_opevals

        point_mut_cost = 0
        _, depth = self.getUpdateOpevals(self.program, 0, self._depth())
        parent = [-1] * len(program)
        for i in range(1, len(program)):
            d = depth[i]
            for j in range(i-1, -1, -1):
                if depth[j] == d-1:
                    parent[i] = j
                    break

        affected_ops = set()
        for node in mutate:
            idx = node
            while idx != -1:
                if isinstance(program[idx], _Function):
                    affected_ops.add(idx)
                idx = parent[idx]

        point_mut_cost = data_size*len(affected_ops)

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        if all_opevals + point_mut_cost >= op_limit:
            point_mut_cost = 0
            genetic.stop = True
            return self.program, list(mutate), point_mut_cost
        else:
            return program, list(mutate), point_mut_cost

        # return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
