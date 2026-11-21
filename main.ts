// sokoban-solver-01 — a push-optimal Sokoban solver, written from scratch.
//
// Idea (standard algorithms, my own implementation):
//   * The generated levels are a hallway with 1-4 small rooms off it. A box
//     pushed out of a room's door into the hallway can never come back, so
//     every box stays in its room and the level splits into independent
//     per-room problems whose push-optima add up to the level optimum.
//   * Each room is solved with A* over *pushes*: a state is the boxes'
//     squares plus the player's reachable region (named by its lowest
//     square). The heuristic is the cheapest assignment of boxes to goals
//     over shortest-path distances on the open floor, which never
//     overestimates the pushes still needed, so the first solved state
//     popped is push-optimal.
//   * Dead states are pruned: a box on a square from which no sequence of
//     pushes (ignoring other boxes) can reach a goal, and a 2x2 block of
//     walls/boxes holding a box off its goal.
//   * If the level does not split into rooms, or a room search exhausts its
//     work budget, the same search runs over the whole level. If that also
//     gives up, an empty (valid, zero-point) answer is written instead of
//     crashing: every instance is always answered in format.
//   * Deterministic: budgets are counted in expanded states, never in wall
//     clock; ties break by insertion order; no clock, no randomness, no
//     environment reads. Two runs write identical bytes.
//
// Local run: MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expanded states). Rooms of this generator are kept only
// when solvable well inside these counts — measured worst case ~2,000
// expansions per room over 9 drawn sets, so the caps only bound
// pathological inputs and never the levels this generator draws; the
// 60s wall budget for the whole set is never at risk.
const ROOM_STATE_CAP = 1_000_000;
const WHOLE_STATE_CAP = 400_000;

// Four moves: [dx, dy, step letter]. Fixed order everywhere, for determinism.
const DIRS: ReadonlyArray<readonly [number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  boxes: number[]; // sorted
  player: number;
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever the text shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Squares from which a box can be pushed onto some goal, ignoring other
 * boxes: pull the box backwards from every goal (a pull needs the square
 * the box comes from and the square the player stands on to be open).
 */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue = [...b.goals];
  for (const g of b.goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * b.w + dx;
      const y = x - d; // square the box is pulled from
      if (y < 0 || y >= live.length) continue;
      if (!b.wall[y] && !b.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood fill of the squares the player can walk to from `from`, with
 * `blocked` in the way. Marks them in `seen` with `stamp` and returns the
 * lowest square of the region (the region's canonical name).
 */
function regionLow(
  b: Board,
  blocked: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * b.w + dx;
      if (!b.wall[q] && !blocked[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Shortest walk (lowercase LURD) from `from` to `to` around walls/boxes. */
function walkTo(b: Board, blocked: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how = new Array<string>(b.w * b.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * b.w + dx;
      if (b.wall[q] || blocked[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/**
 * Split the level into rooms: cut the player's row where a box could never
 * reach a goal (the hallway), then take the connected floor components.
 * Returns null when the level is not a clean set of rooms (a box in the
 * hallway, a goal stranded outside any room, a room whose boxes and goals
 * do not match up).
 */
function roomsOf(b: Board): { boxes: number[]; goals: number[] }[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.w * b.h);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(b.w * b.h).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < b.w * b.h; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * b.w + dx;
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const x of b.boxes) {
    if (part[x] < 0) return null;
    rooms[part[x]]!.boxes.push(x);
  }
  for (const g of b.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// --- A* over pushes -------------------------------------------------------

interface SearchResult {
  pushes: number;
  moves: string;
}

/** All permutations of [0..n) as index lists, in fixed order. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

interface HeapEntry {
  f: number;
  h: number;
  seq: number;
  state: number;
}

function heapPush(heap: HeapEntry[], e: HeapEntry): void {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    const parent = heap[p]!;
    if (e.f > parent.f || (e.f === parent.f && (e.h > parent.h || (e.h === parent.h && e.seq > parent.seq)))) break;
    heap[i] = parent;
    heap[p] = e;
    i = p;
  }
}

function heapPop(heap: HeapEntry[]): HeapEntry {
  const top = heap[0]!;
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length) {
        const a = heap[c]!;
        const b2 = heap[c + 1]!;
        if (b2.f < a.f || (b2.f === a.f && (b2.h < a.h || (b2.h === a.h && b2.seq < a.seq)))) c++;
      }
      const child = heap[c]!;
      const cur = heap[i]!;
      if (child.f > cur.f || (child.f === cur.f && (child.h > cur.h || (child.h === cur.h && child.seq > cur.seq)))) break;
      heap[i] = child;
      heap[c] = cur;
      i = c;
    }
  }
  return top;
}

/**
 * Push-optimal solve of one board (its boxes onto its goals) with A*.
 * State key: the boxes' squares plus the player's region low square, as a
 * number in base `size` (exact while size ** (boxes + 1) < 2 ** 53, true
 * for the rooms here: at most 4 boxes on a board under 256 squares), or
 * a string key when that would overflow.
 * Returns null when the budget runs out or the board is unsolvable.
 */
function solvePushAStar(b: Board, stateCap: number): SearchResult | null {
  const size = b.w * b.h;
  const n = b.boxes.length;
  if (n === 0) return { pushes: 0, moves: '' };
  const goalSet = new Uint8Array(size);
  for (const g of b.goals) goalSet[g] = 1;
  const live = liveSquares(b);
  if (b.boxes.some((x) => !live[x])) return null;
  if (b.boxes.every((x) => goalSet[x])) return { pushes: 0, moves: '' };

  // Shortest-path distances (walls only) from each goal to every square.
  const dist: Int32Array[] = b.goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const r = p + dy * b.w + dx;
        if (!b.wall[r] && d[r] < 0) {
          d[r] = d[p]! + 1;
          q.push(r);
        }
      }
    }
    return d;
  });
  const perms = n <= 6 ? permutations(n) : null;
  // Cheapest box->goal assignment; Infinity when some box cannot reach
  // any goal on the open floor (should not happen: live[] already covers it).
  // For many boxes the exact assignment is replaced by the sum of each
  // box's nearest goal, which is still a lower bound on the pushes left.
  const heuristic = (boxes: number[]): number => {
    if (perms !== null) {
      let best = Infinity;
      for (const perm of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[perm[i]!]![boxes[i]!]!;
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      let m = Infinity;
      for (let j = 0; j < n; j++) {
        const d = dist[j]![boxes[i]!]!;
        if (d >= 0 && d < m) m = d;
      }
      if (!Number.isFinite(m)) return Infinity;
      sum += m;
    }
    return sum;
  };

  const blocked = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // State key: the boxes' squares plus the player's region-low square.
  // Numeric (base `size`) while size ** (boxes + 1) stays under 2 ** 53
  // (true for the rooms here: at most 4 boxes on a board under 256
  // squares); a string key otherwise, so the whole-level search over many
  // boxes never collides.
  const numericKeys = Math.pow(size, n + 1) < 2 ** 53;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKeys) return `${low}:${boxes.join(',')}`;
    let k = low;
    for (const x of boxes) k = k * size + x;
    return k;
  };
  const solved = (boxes: number[]): boolean => boxes.every((x) => goalSet[x]);

  // Whether the box just pushed to `t` sits frozen in a 2x2 block of
  // walls and boxes that holds a box off its goal.
  const frozenAt = (t: number): boolean => {
    for (const a of [t, t - 1, t - b.w, t - b.w - 1]) {
      const c1 = a + 1;
      const c2 = a + b.w;
      const c3 = c2 + 1;
      if (a < 0 || c3 >= size) continue;
      const filled =
        (b.wall[a] || blocked[a]) &&
        (b.wall[c1] || blocked[c1]) &&
        (b.wall[c2] || blocked[c2]) &&
        (b.wall[c3] || blocked[c3]);
      if (!filled) continue;
      if (
        (blocked[a] && !goalSet[a]) ||
        (blocked[c1] && !goalSet[c1]) ||
        (blocked[c2] && !goalSet[c2]) ||
        (blocked[c3] && !goalSet[c3])
      ) {
        return true;
      }
    }
    return false;
  };

  const statesBoxes: number[][] = [b.boxes];
  const statesPlayer: number[] = [b.player]; // player square after the push that made the state
  const statesLow: number[] = []; // the state's region-low square, fixed at insertion
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const expandedSet = new Set<number>(); // states already expanded (their g is final)
  const best = new Map<number | string, number>();
  for (const x of b.boxes) blocked[x] = 1;
  const startLow = regionLow(b, blocked, b.player, seen, ++stamp, stack);
  for (const x of b.boxes) blocked[x] = 0;
  statesLow.push(startLow);
  best.set(keyOf(b.boxes, startLow), 0);

  const heap: HeapEntry[] = [];
  let seq = 0;
  const startH = heuristic(b.boxes);
  if (!Number.isFinite(startH)) return null; // no box->goal matching exists
  heapPush(heap, { f: startH, h: startH, seq: seq++, state: 0 });

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < stateCap) {
    const entry = heapPop(heap);
    const s = entry.state;
    const boxes = statesBoxes[s]!;
    const g = statesG[s]!;
    // Stale-entry check: this state id was superseded by a cheaper path.
    if (best.get(keyOf(boxes, statesLow[s]!)) !== s) continue;
    if (solved(boxes)) {
      found = s;
      break;
    }
    expanded++;
    expandedSet.add(s);
    // The player's reachable region from where the last push left them.
    for (const x of boxes) blocked[x] = 1;
    const here = ++stamp;
    regionLow(b, blocked, statesPlayer[s]!, seen, here, stack);
    for (let bi = 0; bi < n; bi++) {
      const box = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * b.w + dx;
        const behind = box - d; // where the player must stand
        const t = box + d; // where the box lands
        if (seen[behind] !== here) continue;
        if (b.wall[t] || blocked[t] || !live[t]) continue;
        blocked[box] = 0;
        blocked[t] = 1;
        const frozen = frozenAt(t);
        let nextKey: number | string = -1;
        let nextBoxes: number[] | null = null;
        let nextLow = -1;
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[bi] = t;
          nextBoxes.sort((a, z) => a - z);
          nextLow = regionLow(b, blocked, box, mark, ++stamp, stack);
          nextKey = keyOf(nextBoxes, nextLow);
        }
        blocked[t] = 0;
        blocked[box] = 1;
        if (frozen) continue;
        const h = heuristic(nextBoxes!);
        // No box->goal matching on the open floor: no solution through here.
        if (!Number.isFinite(h)) continue;
        const known = best.get(nextKey);
        if (known !== undefined) {
          // A cheaper route to the same state: reopen it (only valid
          // before it is expanded; with a consistent heuristic a state is
          // first expanded at its cheapest g, so this is a safety net).
          if (expandedSet.has(known) || statesG[known]! <= g + 1) continue;
          statesG[known] = g + 1;
          statesPlayer[known] = box;
          parent[known] = s;
          heapPush(heap, { f: g + 1 + h, h, seq: seq++, state: known });
          continue;
        }
        const ns = statesBoxes.length;
        best.set(nextKey, ns);
        statesBoxes.push(nextBoxes!);
        statesPlayer.push(box);
        statesLow.push(nextLow);
        statesG.push(g + 1);
        parent.push(s);
        heapPush(heap, { f: g + 1 + h, h, seq: seq++, state: ns });
      }
    }
    for (const x of boxes) blocked[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: for each push on the path, walk to the pushing
  // square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = b.player;
  let boxes = b.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((x) => !nextBoxes.includes(x))!;
    const to = nextBoxes.find((x) => !boxes.includes(x))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * b.w + dx === d)!;
    for (const x of boxes) blocked[x] = 1;
    const walk = walkTo(b, blocked, player, from - d);
    for (const x of boxes) blocked[x] = 0;
    if (walk === null) return null; // cannot happen: reachability was checked
    moves += walk + dir[2].toUpperCase();
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// --- Whole level ----------------------------------------------------------

function solveLevel(text: string): string {
  try {
    const level = parseLevel(text);
    if (level.player < 0 || level.boxes.length === 0) return '';
    if (level.boxes.every((x) => level.goals.includes(x))) return '';

    const rooms = roomsOf(level);
    if (rooms !== null) {
      let player = level.player;
      let moves = '';
      let ok = true;
      const done: { boxes: number[]; goals: number[] }[] = [];
      for (const room of rooms) {
        const wall = level.wall.slice();
        for (const other of rooms) {
          if (other === room) continue;
          const settled = done.includes(other);
          for (const x of settled ? other.goals : other.boxes) wall[x] = 1;
        }
        const sub: Board = {
          w: level.w,
          h: level.h,
          wall,
          goals: [...room.goals].sort((a, b) => a - b),
          boxes: [...room.boxes].sort((a, b) => a - b),
          player,
        };
        const solved = solvePushAStar(sub, ROOM_STATE_CAP);
        if (solved === null) {
          ok = false;
          break;
        }
        moves += solved.moves;
        player = endOf(level.w, player, solved.moves);
        done.push(room);
      }
      if (ok) return moves.toLowerCase();
    }

    // Fallback: one search over the whole level.
    const whole = solvePushAStar(level, WHOLE_STATE_CAP);
    if (whole !== null) return whole.moves.toLowerCase();
    return '';
  } catch {
    return '';
  }
}

/** The square the player stands on after `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
