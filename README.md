# dijkstra algorithm python: heapq implementation, path reconstruction, and the edge cases that break it

You hit a problem with weighted edges, typed `dijkstra` into Google, and got forty blog posts with the same twelve lines of code and three explanations that contradict each other about what's actually in the priority queue. Let's fix that properly.

Below is a Dijkstra implementation in Python that works, plus the parts most tutorials skip: reconstructing the path, why stale heap entries are fine, where the O((V+E) log V) actually comes from, and which inputs will silently give you a wrong answer.

## What the algorithm is doing, in Python terms

Dijkstra answers one question: starting from a source node, what's the cheapest way to reach every other node in a graph with non-negative edge weights?

The mechanism is greedy. You repeatedly take the unvisited node with the smallest known distance, then "relax" its outgoing edges — check whether going through this node gives any neighbor a cheaper route than whatever you had before. Once a node comes out of the priority queue for the first time, its distance is final. That's the whole algorithm, and it's why negative weights break it: the greedy assumption falls apart the moment a later edge can reduce a distance you already locked in.

In Python, the standard way to represent the graph is an adjacency list — a dict mapping each node to a list of `(neighbor, weight)` pairs:

python
graph = {
    "A": [("B", 5), ("C", 3)],
    "B": [("C", 1), ("D", 2)],
    "C": [("D", 6), ("E", 4)],
    "D": [("E", 1)],
    "E": [],
}


Include every node as a key, even the ones with no outgoing edges. Forgetting that is the source of the most common `KeyError` in hand-written Dijkstra code.

## The heapq implementation

`heapq` gives you a binary min-heap, which is exactly the priority queue Dijkstra needs. Push `(distance, node)` tuples and pop the smallest.

python
import heapq

def dijkstra(graph, source):
    dist = {node: float("inf") for node in graph}
    prev = {node: None for node in graph}
    dist[source] = 0

    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)

        # Stale entry: we already found a shorter path to u
        if d > dist[u]:
            continue

        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev


Two details worth understanding rather than memorizing:

**You don't need to pre-insert every vertex.** The MIT-style pseudocode builds a priority queue containing all V vertices and then calls `decrease_key`. Python's `heapq` has no `decrease_key`, and you don't need one. Start the heap with just the source and push a new entry whenever you improve a distance. The heap grows, but each entry is cheap.

**The `if d > dist[u]: continue` line is not optional.** It's how you skip entries that were valid when pushed but superseded later. More on that below.

## Reconstructing the path, not just the distance

`dist` tells you the cost. Interviewers usually want the route. That's what `prev` is for — each time you improve a node's distance, record which node got you there. Then walk backwards from the target:

python
def shortest_path(graph, source, target):
    dist, prev = dijkstra(graph, source)

    if dist[target] == float("inf"):
        return None, float("inf")   # unreachable

    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]

    return path[::-1], dist[target]


The `prev[source] = None` initialization is what terminates the walk. If you initialize `prev` to `None` for all nodes and forget to think about it, you'll get an infinite loop on any graph where the source is reachable from itself in a cycle... except it isn't, because the source's `prev` stays `None` and the loop stops there. Just make sure you don't accidentally set `prev[source]` during relaxation — you won't, since `dist[source]` is 0 and no non-negative edge can produce a smaller value.

One more thing this pattern handles correctly: the unreachable case. Nodes in a disconnected component keep `float("inf")`. Returning `None` for those is better than returning `[]` or crashing, and it's the kind of thing that turns a "correct" solution into a failed test case.

## Why stale heap entries are fine

This trips up almost everyone at some point. Suppose `C` is pushed with distance 7, then improved to 3, then pushed again. The heap now contains both `(7, "C")` and `(3, "C")`. `(3, "C")` pops first, so `dist["C"]` becomes 3. Later, `(7, "C")` pops, and `7 > dist["C"]` is true — so you skip it and move on.

That's lazy deletion, and it's the reason the implementation doesn't need a decrease-key operation. The cost is that the heap can hold up to E entries instead of V, which is where the log factor in the complexity comes from. It's a completely standard trade-off, and it's what production libraries do too.

The alternative — scanning a list for the minimum each iteration, like the textbook "unvisited set" version — is O(V²) and fine for dense graphs. For anything sparse, the heap wins.

## Complexity, and where the number comes from

With an adjacency list and a binary heap:

- **Time: O((V + E) log V)**. Each edge can trigger at most one push. Each heap operation is O(log V), and the heap can hold up to O(E) entries.
- **Space: O(V + E)** for the adjacency list, distances, and predecessors, plus O(E) for the heap in the worst case. If you only care about distances, the working space beyond the graph itself is O(V).

Compare that with BFS on an unweighted graph — O(V + E), no heap — or Bellman-Ford's O(V·E). If your edges all cost the same, Dijkstra is doing unnecessary work. Just use BFS.

## Edge cases that silently break things

Four things to check before you trust your own solution:

**Negative weights.** Dijkstra is wrong here, no exceptions. It won't necessarily crash — it'll return a plausible-looking answer that's too large. Use Bellman-Ford, or SPFA if you know there are no negative cycles.

**Unorderable node types.** `heapq` compares tuples element by element. If two distances tie, Python moves on to comparing the second element. Integers and strings compare fine; custom objects don't, and you'll get a `TypeError` at runtime on a graph that worked yesterday. The fix is a tie-breaking counter: push `(distance, count, node)` with an `itertools.count()`.

**Missing adjacency keys.** `graph[u]` raises `KeyError` for any node that appears only as an edge target. Either build the adjacency list with `graph.setdefault(v, [])` during construction, or use `graph.get(u, [])` in the loop.

**Early exit.** If you only need one target, `break` the moment it's popped from the heap. At that point its distance is final, and you can skip the rest of the graph. Don't break when you push it — pushing means "candidate," popping means "settled."

python
while heap:
    d, u = heapq.heappop(heap)
    if d > dist[u]:
        continue
    if u == target:
        break        # dist[target] is now final
    ...


## Grid problems wear the same hat

A large share of interview questions that need Dijkstra aren't framed as graphs at all — they're 2D grids where each cell has a cost. Same algorithm, distance stored in a 2D array instead of a dict:

python
import heapq

def min_cost_path(grid):
    R, C = len(grid), len(grid[0])
    dist = [[float("inf")] * C for _ in range(R)]
    dist[0][0] = 0
    heap = [(0, 0, 0)]   # (cost, row, col)

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        if r == R - 1 and c == C - 1:
            return d
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < R and 0 <= nc < C:
                nd = d + grid[nr][nc]
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return -1


Two variations are worth knowing because they change the shape of the state:

- **Cost belongs to entering a cell, not to an edge.** Same code, you just read the weight from the destination cell.
- **One edge type is discounted or capped** (a "k stops" constraint, a free pass you can use once). The trick is to put that extra dimension into the node itself — `(cost, r, c, pass_used)` — and run Dijkstra on the expanded graph. This is how the cheapest-flights-within-k-stops family of problems gets solved without Bellman-Ford.

For production code, don't hand-roll any of this. `networkx.dijkstra_path` and `scipy.sparse.csgraph.dijkstra` are both battle-tested and handle the tie-breaking and staleness bookkeeping for you. Write your own version to learn the mechanics and to pass interviews; use a library when the code has to survive contact with real data.

## Practicing this without grinding at random

The reason Dijkstra feels slippery is that the interesting part isn't the code — it's recognizing when a problem is a shortest-path problem in disguise: a state space that isn't obviously a graph, a cost that isn't obviously a weight, a constraint that needs to live inside the node. That recognition is a pattern-matching skill, and it comes from seeing a dozen variations rather than memorizing one template.

This is the case AlgoMonster is built around. It's a text-and-diagram course by ex-Google engineers organized into 48 patterns, with graph and shortest-path material covered alongside the templates for BFS, DFS, and topological sort — so the Dijkstra code above sits next to the problems where you'd actually reach for it, in Python, Java, C++, or whichever language you pick. There are no videos, which is either efficient or annoying depending on how you learn. Lessons, illustrations, and editorial explanations are unlocked on Pro, along with unlimited Speedrun sessions and the AI assistant.

Current plans, as listed on their site:

| Plan | What you get | Price | Billing | Get it |
| --- | --- | --- | --- | --- |
| Free | Limited access to a subset of lessons and problems | $0 | — | [Try the free tier first](https://algo.monster?fpr=li69) |
| Pro Yearly | All patterns, lessons, and problems; unlimited AI assistant; unlimited Speedrun; 500+ lessons across 6 courses; 1,000+ illustrations; 3,000+ editorials; company questions updated monthly; member discount on live bootcamps | $99/year ($8.25/mo billed annually) | Annual, cancel anytime | [Check the Pro Yearly plan](https://algo.monster/subscribe?fpr=li69) |
| Pro Lifetime | Everything in Pro Yearly plus lifetime updates — no renewal | $189 | One-time | [See the Pro Lifetime option](https://algo.monster/subscribe?fpr=li69) |

A few notes on the numbers, because they move around. There's no monthly plan — annual or lifetime only. The yearly figure works out to roughly 28 cents a day, and the lifetime price is what you'll see on the checkout page at the moment you read this; AlgoMonster runs sales often enough that third-party blogs quote anything from $89 to $459, so trust the page over the blogs, including this one. Both paid tiers include the bonus CS fundamentals and "build an LLM from scratch" courses. Specialized partner content, such as Quant Interview Prep, is billed separately.

Is it worth it? If your gap is "I can write Dijkstra but I don't notice when a problem needs it," structured pattern training addresses exactly that, and $99 for a year is less than most people spend on coffee during interview prep. If you already recognize the patterns and only need volume, a big problem bank with a monthly subscription may fit you better. The free tier is the honest test — if the text-and-diagram format doesn't click for you in an afternoon, the paid plans won't fix that.

## The short version

Use `heapq`, push `(distance, node)` pairs, skip stale pops with a `dist` comparison, track predecessors if you need the path, and remember that a popped node's distance is final. Keep negative weights and unorderable node types away from it. When a problem hands you a grid or a state space instead of a graph, the code barely changes — only the node definition does. That last part is the skill worth practicing, because the code itself you'll have written correctly within an hour.

👉 [Start with the free AlgoMonster tier and see how the graph patterns are laid out](https://algo.monster?fpr=li69)
