#!/usr/bin/env python3
# Student name: NAME
# Student number: NUMBER
# UTORid: ID

import typing as T
from math import inf

import torch
from torch.nn.functional import pad
from torch import Tensor


def is_projective(heads: T.Iterable[int]) -> bool:
    """
    Determines whether the dependency tree for a sentence is projective.

    Args:
        heads: The indices of the heads of the words in sentence. Since ROOT
          has no head, it is not expected to be part of the input, but the
          index values in heads are such that ROOT is assumed in the
          starting (zeroth) position. See the examples below.

    Returns:
        True if and only if the tree represented by the input is
          projective.

    Examples:
        The projective tree from the assignment handout:
        >>> is_projective([2, 5, 4, 2, 0, 7, 5, 7])
        True

        The non-projective tree from the assignment handout:
        >>> is_projective([2, 0, 2, 2, 6, 3, 6])
        False
    """
    projective = True
    # *** BEGIN YOUR CODE *** #
    # Graph is not projective iff two 'zones' overlap
    for i in range(len(heads)):
        head_index = heads[i]-1  # So we create an interval [i, head_index]
        if head_index < i: # zone is on the left of i
            upper = i
            lower = head_index
        else:             # zone is on the right of i
            upper = head_index
            lower = i

        for j in range(len(heads)):
            j_head_index = heads[j]-1
            # if j falls inside the zone yet its head falls outside, 
            # or if j's head is inside the zone but j is outside:
            if (lower < j < upper and (j_head_index < lower or j_head_index > upper)) or \
            (lower < j_head_index < upper and (j < lower or j > upper)):
                 projective = False
                 break
    # *** END YOUR CODE *** #
    return projective


def is_single_root_tree(heads: Tensor, lengths: Tensor) -> Tensor:
    """
    Determines whether the selected arcs for a sentence constitute a tree with
    a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    This file already imports the function `pad` for you. You may find that
    function handy. Here's the documentation of the function:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    Args:
        heads (Tensor): a Tensor of dimensions (batch_sz, sent_len) and dtype
            int where the entry at index (b, i) indicates the index of the
            predicted head for vertex i for input b in the batch

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype bool and dimensions (batch_sz,) where the value
        for each element is True if and only if the corresponding arcs
        constitute a single-root-word tree as defined above

    Examples:
        Valid trees from the assignment handout:
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 7, 5, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 7]))
        tensor([True, True])

        Invalid trees (the first has a cycle; the second has multiple roots):
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 8, 6, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 8]))
        tensor([False, False])
    """
    # *** BEGIN YOUR CODE *** #
    tree_single_root = torch.ones_like(heads[:, 0], dtype=torch.bool)
    # A tree is a connected acyclic graph

    def check_acyclic(heads, lengths, batch_index):
        """
        Returns True if there is no cycle in the graph in batch_index, False otherwise
        """
        # Initiate cycle checking for every node
        for i in range(lengths[batch_index]):
            cur = i
            seen = [cur]
            counter = 0
            while counter < lengths[batch_index]: # We can draw at most 'length' arcs from dependents to heads
                cur = heads[batch_index, cur]-1  # find the head of cur, rename it as the new cur
                if cur == -1:  # if cur is the ROOT, then there is no cycle
                    break
                if cur in seen:  # if cur is also a dependent we have seen before, there is a cycle
                    return False
                # Otherwise add this head to the seen list, and it becomes the new dependent for the next iteration
                seen.append(cur)
                counter += 1
        return True


    # For every sentence in the batch
    for i in range(len(heads)):
        # Check if there are cycles
        if not check_acyclic(heads, lengths, i):
            tree_single_root[i] = False
            continue

        # Check if tree is connected (i.e. need to check for every element, 
        # either it is the head of something, or it is the dependent of something)
        # However, by construction every node already has a head
            

        # Check if there are multiple roots
        root_counter = 0
        for j in range(lengths[i]):  # Length inconsistent in heads, always use the lengths tensor
            if heads[i, j] == 0:
                root_counter += 1
            if root_counter > 1:
                tree_single_root[i] = False
                break
    # *** END YOUR CODE *** #
    return tree_single_root



#############################################
# Prim's algorithm for MST
# Code adapted from https://www.geeksforgeeks.org/maximum-spanning-tree-using-prims-algorithm/

def findMaxVertex(visited, weights, V):
    """
    Find the max weight vertex from the set of unvisited vertices
    """
    
    index = -1   # index of highest-weight vertex fromt he set of unvisited vertices
    maxW = -inf   # max weight in the set of unvisited vertices
 
    # for each node
    for i in range(V):
        # If the current node is not visited and weight of current vertex is
        # greater than the max weight we have seen
        if (visited[i] == False and weights[i] > maxW):
            maxW = weights[i] # Update maxW
            index = i         # Update index
    return index
 
 
# Function to find the maximum spanning tree
def maximumSpanningTree(graph, V):
    """
    Construct a MST using Prim's algorithm, noting that for each max weight vertex we select,
    we must not violate the single root constraint.
    """
    visited = [True]*V  # stores whether each vertex has been visited
    weights = [0]*V     # weights[i]: The max weight of connecting an edge with vertex i
    parent = [0]*V      # parent[i]: The parent of vertex i in the MST
 
    # Initialize weights and visited
    for i in range(V):
        visited[i] = False
        weights[i] = -inf
 
    # include 1st vertex (root) in maximum spanning tree
    weights[0] = max(graph[:, 0])  # The max value to be eaten by the root
    parent[0] = 0  # By deaulft, root's parent is 0
 
    # Search for other (V-1) vertices greedily
    for i in range(V - 1):
        maxVertexIndex = findMaxVertex(visited, weights, V)  # Find the most valuable unexplored vertex
 
        visited[maxVertexIndex] = True  # Mark that vertex as visited
 
        if maxVertexIndex != 0:  # If we are not selecting the root
            for j in range(V):   # Explore vertices j that are adjacent to maxVertexIndex
    
                # If there is an edge from current visited vertex to j, and j is unvisited
                if (graph[j][maxVertexIndex] != 0 and visited[j] == False):
                    
                    # See if we can draw an arc from head to dependent
                    # NOTE: if the head is 0, we must be cautious and
                    # greedily select the best j to be its dependent

                    # If edge weight is better than what we have seen
                    if (graph[j][maxVertexIndex] > weights[j]):
                        
                        # Update weights[j]
                        weights[j] = graph[j][maxVertexIndex] 

                        # Update parent[j]
                        parent[j] = maxVertexIndex        

                if (graph[maxVertexIndex][j] != 0 and visited[j] == False):
                    if (graph[maxVertexIndex][j] > weights[j]):
                        weights[j] = graph[maxVertexIndex][j]
                        parent[j] = maxVertexIndex
        
        # Else, we can only have 1 arc from 
        # ROOT to a dependent
        else:                   # Note, since we are slecting the unvisited vertices, ROOT will only appear once
            max_j = None
            max_weight = -inf
            for j in range(V):
                if (graph[j][0] != 0 and visited[j] == False):  # if ROOT can draw arc to j, and j is unvisited
                    if (graph[j][0] > weights[j]):              # if j is better than the current best for j
                        if graph[j][0] > max_weight:            # if j beats the current max weight
                            max_weight = graph[j][0]
                            max_j = j
            if max_j is not None:
                weights[max_j] = max_weight
                parent[max_j] = 0

 
    # Print maximum spanning tree
    return parent

# def msa(V, E, r, w):
#     """
#     Recursive Edmond's algorithm as per Wikipedia's explanation
#     Returns a set of all the edges of the minimum spanning arborescence.
#     V := set( Vertex(v) )
#     E := set( Edge(u,v) )
#     r := Root(v)
#     w := dict( Edge(u,v) : cost)
#     """

#     """
#     Step 1 : Removing all edges that lead back to the root
#     """
#     for (u,v) in E.copy():
#         if v == r:
#             E.remove((u,v))
#             w.pop((u,v))

#     """
#     Step 2 : Finding the max incoming edge for every vertex. 
#     """
#     pi = {}
#     for v in V:
#         max_edge = None
#         max_weight = -inf
#         for u,v in E:
#             if v == v:
#                 if w[(u,v)] > max_weight:
#                     max_edge = (u,v)
#                     max_weight = w[(u,v)]
#         pi[v] = max_edge[0]
    
#     """
#     Step 3 : Finding cycles in the set of edges {(pi[v],v), v in V\{r})}
#     """
#     cycle_vertex = None
#     for v in V:
#         if v == r:
#             continue
#         if cycle_vertex is not None:
#             break
#         u = pi.get(v)
#         seen = set()
#         seen.add(u)
#         while u != r:
#             u = pi.get(u)  # update u until we reach the root (i.e. root is the parent of u)
#             if u in seen:
#                 cycle_vertex = u
#                 break
#             seen.add(u)

    
#     """
#     Step 4 : If there is no cycle, then the spanning tree is defined by 
#     the set of edges {(pi[v],v), v in V\{r})}
#     """
#     if cycle_vertex is None:
#         max_edges = []
#         for u,v in pi.items():
#             max_edges.append((v,u))

        
    
#     """
#     Step 5 : Otherwise, all the vertices in the cycle must be identified
#     """
#     C = set()
#     C.add(cycle_vertex)
#     parent = pi.get(cycle_vertex)
#     while parent != cycle_vertex:
#         C.add(parent)
#         parent = pi[parent]
    
#     """
#     Step 6 : Contracting the cycle C into a new node v_c
#     Define a new weighted directed graph D_prime = (V_prime, E_prime, w_prime)
#     """
#     v_c = 0.5  # Use a float to avoid collision with other vertices
#     V_prime = set()
#     E_prime = set()
#     w_prime = {}
#     correspondance = {}  # for each edge in E', we remember which edge in the original graph it corresponds to
#     for v in V:
#         if v not in C:
#             V_prime.add(v)

#     for u,v in E:
#         # if it is an incoming edge to the cycle
#         if u not in C and v in C:
#             e = (u, v_c)
#             w_prime[e] = w[(u,v)] - w[(pi[v],v)]
#             E_prime.add(e)
#             correspondance[e] = (u,v)

#         # if it is an outgoing edge from the cycle
#         elif u in C and v not in C:
#             e = (v_c, v)
#             w_prime[e] = w[(u,v)]
#             E_prime.add(e)
#             correspondance[e] = (u,v)

#         # if it is an edge that's unrelated to the cycle
#         elif u not in C and v not in C:
#             e = (u,v)
#             w_prime[e] = w[(u,v)]
#             E_prime.add(e)
#             correspondance[e] = (u,v)
    
#     """
#     Step 7 : Recursively calling the algorithm again until no cycles are found
#     """
#     tree = msa(V_prime, E_prime, r, w_prime)
    
#     """
#     Step 8 : 
#     """
#     cycle_edge = None
#     for (u,v) in tree:
#         if v == v_c:
#             old_v = correspondance[(u,v_c)][1]
#             cycle_edge = (pi[old_v],old_v)
#             break
    
#     ret = set([correspondance[(u,v)] for (u,v) in tree])
#     for v in C:
#         u = pi[v]
#         ret.add((u,v))
        
#     ret.remove(cycle_edge)
    
#     return ret


def msa(V, E, r, w):
    """
    Recursive Edmond's algorithm as per Wikipedia's explanation
    Returns a set of all the edges of the minimum spanning arborescence.
    V := set( Vertex(v) )
    E := set( Edge(u,v) )
    r := Root(v)
    w := dict( Edge(u,v) : cost)
    """
    
    """
    Step 1 : Removing all edges that lead back to the root
    """
    for (u,v) in E.copy():
        if v == r:
            E.remove((u,v))
            w.pop((u,v))

    """
    Step 2 : Finding the minimum incoming edge for every vertex.
    """
    pi = dict()
    for v in V:
        edges = [edge[0] for edge in E if edge[1] == v]
        if not len(edges):
            continue
        costs = [w[(u,v)] for u in edges]
        pi[v] = edges[costs.index(min(costs))]
    
    """
    Step 3 : Finding cycles in the graph
    """
    cycle_vertex = None
    for v in V:
        if cycle_vertex is not None:
            break
        visited = set()
        next_v = pi.get(v)
        while next_v:
            if next_v in visited:
                cycle_vertex = next_v
                break
            visited.add(next_v)
            next_v = pi.get(next_v)
    
    """
    Step 4 : If there is no cycle, return all the minimum edges pi(v)
    """
    if cycle_vertex is None:
        return set([(pi[v],v) for v in pi.keys()])
    
    """
    Step 5 : Otherwise, all the vertices in the cycle must be identified
    """
    C = set()
    C.add(cycle_vertex)
    next_v = pi.get(cycle_vertex)
    while next_v != cycle_vertex:
        C.add(next_v)
        next_v = pi.get(next_v)
    
    """
    Step 6 : Contracting the cycle C into a new node v_c
    v_c is negative and squared to avoid having the same number
    """
    v_c = -cycle_vertex**2
    V_prime = set([v for v in V if v not in C] + [v_c])
    E_prime = set()
    w_prime = dict()
    correspondance = dict()
    for (u,v) in E:
        if u not in C and v in C:
            e = (u,v_c)
            if e in E_prime:
                if w_prime[e] < w[(u,v)] - w[(pi[v],v)]:
                    continue
            w_prime[e] = w[(u,v)] - w[(pi[v],v)]
            correspondance[e] = (u,v)
            E_prime.add(e)
        elif u in C and v not in C:
            e = (v_c,v)
            if e in E_prime:
                old_u = correspondance[e][0]
                if w[(old_u,v)] < w[(u,v)]:
                    continue
            E_prime.add(e)
            w_prime[e] = w[(u,v)]
            correspondance[e] = (u,v)
        elif u not in C and v not in C:
            e = (u,v)
            E_prime.add(e)
            w_prime[e] = w[(u,v)]
            correspondance[e] = (u,v)
    
    """
    Step 7 : Recursively calling the algorithm again until no cycles are found
    """
    tree = msa(V_prime, E_prime, r, w_prime)
    
    """
    Step 8 : 
    """
    cycle_edge = None
    for (u,v) in tree:
        if v == v_c:
            old_v = correspondance[(u,v_c)][1]
            cycle_edge = (pi[old_v],old_v)
            break
    
    ret = set([correspondance[(u,v)] for (u,v) in tree])
    for v in C:
        u = pi[v]
        ret.add((u,v))
        
    ret.remove(cycle_edge)
    
    return ret





def single_root_mst(arc_scores: Tensor, lengths: Tensor) -> Tensor:
    """
    Finds the maximum spanning tree (more technically, arborescence) for the
    given sentences such that each tree has a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    This file already imports the function `pad` for you. You may find that
    function handy. Here's the documentation of the function:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    Args:
        arc_scores (Tensor): a Tensor of dimensions (batch_sz, x, y) and dtype
            float where x=y and the entry at index (b, i, j) indicates the
            score for a candidate arc from vertex j to vertex i.

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype int and dimensions (batch_sz, x) where the value at
        index (b, i) indicates the head for vertex i according to the
        maximum spanning tree for the input graph.

    Examples:
        >>> single_root_mst(torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]]),\
            torch.tensor([3, 2, 3]))
        tensor([[0, 0, 3, 1],
                [0, 2, 0, 0],
                [0, 2, 0, 2]])
    """
    # *** BEGIN YOUR CODE *** #
    # best_arcs = arc_scores.argmax(-1)  # compute maximum along the columns of each row, set its index into a tensor
    # return best_arcs
    # print(best_arcs)
    # return is_single_root_tree(best_arcs[:, 1:], lengths)

    best_arcs = torch.zeros_like(arc_scores[:, :, 0], dtype=int)

    # For every sentence in the batch
    for b in range(len(arc_scores)):
        # Create a graph matrix
        graph = arc_scores[b]
        V = range(lengths[b] + 1)  # +1 for the ROOT
        E = set()
        w = dict()

        # To satisfy single root, find the highest weight arc from ROOT to a dependent
        cur_best_v = None
        cur_best_w = -inf

        # For each depdendent v,
        # find all edges
        for v in V: 
            
            # find the best v from ROOT to v

            if v != 0 and graph[v, 0] > cur_best_w:
                cur_best_v = v
                cur_best_w = graph[v, 0]

            # For each head u
            for u in V:
                if u == 0:
                    continue
                if u != v and graph[v, u] != 0:  # if there's an edge from u to v
                    E.add((u, v))
                    w[(u, v)] = -graph[v, u]


        # Add the best edge from 
        # ROOT to a dependent

        E.add((0, cur_best_v))
        w[(0, cur_best_v)] = -cur_best_w

        
        # Compute the MST
        edges = msa(V, E, 0, w)
        for (u, v) in edges:
            best_arcs[b, v] = u

    # *** END YOUR CODE *** #
    return best_arcs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
