# Sokoban solver

Push-optimal Sokoban solver for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121).

The generated levels are a hallway with small rooms off it; a box pushed out
of a room can never return, so each room is an independent problem. Every
room is solved push-optimally with A* over pushes (assignment heuristic,
deadlock pruning), and the room solutions are joined with hallway walks.
On the kit's practice set and 8 freshly drawn sets it scores 1,000,000 —
every instance in exactly its push-optimal count — in under half a second
for a whole set.

Run: `node main.ts` with `/input/<id>/level.txt` in place; it writes
`/output/<id>/solution.txt` (or set `MAKEFUN_INPUT_DIR` /
`MAKEFUN_OUTPUT_DIR` for a local run). Node 22+, no dependencies.

License: MIT.
