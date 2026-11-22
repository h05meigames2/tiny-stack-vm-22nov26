// Tiny stack VM: my own interpreter for make.fun's tiny stack ISA v1.
// One program answers a whole set: for each /input/<id>/ it runs
// program.bin on input.bin and writes /output/<id>/stdout.bin + exit.txt.
// All arithmetic wraps to 32 bits; every trap keeps the output written
// before it and exits with the trap's own code.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STEPS = 1_000_000;
const OUT_MAX = 65_536;

// [opcode, immediate bytes, values popped, values pushed]
const SPEC: Array<[number, number, number, number]> = [
  [0x00, 0, 0, 0], [0x01, 1, 0, 1], [0x02, 4, 0, 1], [0x03, 0, 1, 0],
  [0x04, 0, 1, 2], [0x05, 0, 2, 2], [0x06, 0, 2, 3], [0x07, 0, 3, 3],
  [0x10, 0, 2, 1], [0x11, 0, 2, 1], [0x12, 0, 2, 1], [0x13, 0, 2, 1],
  [0x14, 0, 2, 1], [0x15, 0, 2, 1], [0x16, 0, 2, 1], [0x17, 0, 1, 1],
  [0x18, 0, 2, 1], [0x19, 0, 2, 1], [0x1a, 0, 2, 1], [0x1b, 0, 1, 1],
  [0x1c, 0, 2, 1], [0x1d, 0, 2, 1], [0x1e, 0, 2, 1], [0x20, 0, 2, 1],
  [0x21, 0, 2, 1], [0x22, 0, 2, 1], [0x30, 2, 0, 0], [0x31, 2, 1, 0],
  [0x32, 2, 1, 0], [0x33, 2, 0, 0], [0x34, 0, 0, 0], [0x40, 0, 1, 1],
  [0x41, 0, 2, 0], [0x50, 0, 1, 0], [0x51, 0, 1, 0], [0x52, 0, 0, 1],
  [0x53, 0, 1, 0],
];
const IMM = new Uint8Array(256);
const POPN = new Uint8Array(256);
const PUSHN = new Uint8Array(256);
const KNOWN = new Uint8Array(256);
for (const [op, im, po, pu] of SPEC) {
  IMM[op] = im; POPN[op] = po; PUSHN[op] = pu; KNOWN[op] = 1;
}

function runVM(prog: Uint8Array, input: Uint8Array): { out: Uint8Array; exit: number } {
  const n = prog.length;
  const st = new Int32Array(256); // data stack
  const cs = new Int32Array(64); // call stack
  const mem = new Int32Array(256);
  const out = new Uint8Array(OUT_MAX);
  let sp = 0, cp = 0, pc = 0, ip = 0, steps = 0, olen = 0;
  const done = (code: number): { out: Uint8Array; exit: number } => ({ out: out.slice(0, olen), exit: code });
  for (;;) {
    if (steps === STEPS) return done(10); // step limit
    if (pc < 0 || pc >= n) return done(9); // pc outside the program
    const op = prog[pc];
    if (!KNOWN[op]) return done(5); // unknown opcode
    const imm = IMM[op];
    if (pc + imm >= n) return done(9); // immediate cut off by the end
    steps++;
    const next = pc + 1 + imm;
    if (sp < POPN[op]) return done(2); // stack underflow
    if (sp - POPN[op] + PUSHN[op] > 256) return done(3); // stack overflow
    pc = next;
    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: st[sp++] = (prog[pc - 1] << 24) >> 24; break; // PUSH8, sign-extended
      case 0x02: // PUSH32, little-endian
        st[sp++] = prog[pc - 4] | (prog[pc - 3] << 8) | (prog[pc - 2] << 16) | (prog[pc - 1] << 24);
        break;
      case 0x03: sp--; break; // POP
      case 0x04: st[sp] = st[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = st[sp - 1]; st[sp - 1] = st[sp - 2]; st[sp - 2] = t; break; } // SWAP
      case 0x06: st[sp] = st[sp - 2]; sp++; break; // OVER
      case 0x07: { // ROT
        const a = st[sp - 3]; st[sp - 3] = st[sp - 2]; st[sp - 2] = st[sp - 1]; st[sp - 1] = a;
        break;
      }
      case 0x17: st[sp - 1] = (-st[sp - 1]) | 0; break; // NEG, wraps
      case 0x1b: st[sp - 1] = ~st[sp - 1]; break; // NOT
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const b = st[sp - 1];
        const a = st[sp - 2];
        let r: number;
        if (op === 0x10) r = (a + b) | 0;
        else if (op === 0x11) r = (a - b) | 0;
        else if (op === 0x12) r = Math.imul(a, b); // exact low 32 bits
        else if (op === 0x13) { if (b === 0) return done(4); r = (a / b) | 0; } // trunc toward zero; INT_MIN/-1 wraps
        else if (op === 0x14) { if (b === 0) return done(4); r = (a % b) | 0; } // sign of a; x%-1 is 0
        else if (op === 0x15) { if (b === 0) return done(4); r = Math.floor((a >>> 0) / (b >>> 0)) | 0; }
        else if (op === 0x16) { if (b === 0) return done(4); r = ((a >>> 0) % (b >>> 0)) | 0; }
        else if (op === 0x18) r = a & b;
        else if (op === 0x19) r = a | b;
        else if (op === 0x1a) r = a ^ b;
        else if (op === 0x1c) r = a << (b & 31);
        else if (op === 0x1d) r = a >> (b & 31);
        else if (op === 0x1e) r = (a >>> (b & 31)) | 0;
        else if (op === 0x20) r = a === b ? 1 : 0;
        else if (op === 0x21) r = a < b ? 1 : 0;
        else r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU
        sp -= 2;
        st[sp++] = r;
        break;
      }
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ
        const rel = ((prog[pc - 2] | (prog[pc - 1] << 8)) << 16) >> 16;
        let take = op === 0x30;
        if (op !== 0x30) { const v = st[--sp]; take = op === 0x31 ? v === 0 : v !== 0; }
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL
        if (cp === 64) return done(7);
        cs[cp++] = pc;
        pc = prog[pc - 2] | (prog[pc - 1] << 8);
        break;
      case 0x34: // RET
        if (cp === 0) return done(8);
        pc = cs[--cp];
        break;
      case 0x40: { // LOAD
        const addr = st[sp - 1];
        if (addr < 0 || addr > 255) return done(6);
        st[sp - 1] = mem[addr];
        break;
      }
      case 0x41: { // STORE (v addr --)
        const addr = st[sp - 1];
        const v = st[sp - 2];
        if (addr < 0 || addr > 255) return done(6);
        sp -= 2;
        mem[addr] = v;
        break;
      }
      case 0x50: { // PRINT: decimal ASCII, no newline; digits written straight
        // into out (no string/TextEncoder allocation per print)
        let v = st[--sp];
        const neg = v < 0;
        if (neg) v = -v; // exact in doubles even for -2147483648
        const dig = new Uint8Array(10); // digits of |v|, least significant first
        let tl = 0;
        do { dig[tl++] = 48 + (v % 10); v = (v / 10) | 0; } while (v > 0);
        if (olen + tl + (neg ? 1 : 0) > OUT_MAX) return done(11);
        if (neg) out[olen++] = 45; // '-'
        while (tl > 0) out[olen++] = dig[--tl];
        break;
      }
      case 0x51: // PUTC: one byte, v AND 255
        if (olen + 1 > OUT_MAX) return done(11);
        out[olen++] = st[--sp] & 255;
        break;
      case 0x52: st[sp++] = ip < input.length ? input[ip++] : -1; break; // IN
      case 0x53: return done(st[--sp] & 255); // EXIT: low byte
    }
  }
}

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(IN_DIR).sort()) {
  const prog = readFileSync(join(IN_DIR, id, 'program.bin'));
  const input = readFileSync(join(IN_DIR, id, 'input.bin'));
  const { out, exit } = runVM(new Uint8Array(prog), new Uint8Array(input));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'stdout.bin'), out);
  writeFileSync(join(OUT_DIR, id, 'exit.txt'), `${exit}\n`);
}
