# 🔍 Bitcoin Puzzle Searcher (GPU with Transformations)

This project is a **GPU-accelerated Bitcoin puzzle solver** using CUDA and a unique transformation-based algorithm. It applies a wide variety of bitwise transformations to explore the Bitcoin keyspace in a highly parallelized manner using thousands of GPU threads.

---

## 🚀 How to Use

### Syntax

```
main.exe <start_range> <end_range> <target_hash160> <blocks> <threads>
```

### Examples

```
main.exe 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56 32 32      # Puzzle 21
main.exe 1000000 1ffffff 2f396b29b27324300d0c59b17c3abc1835bd3dbb 32 32     # Puzzle 25
main.exe 100000000 1ffffffff 4e15e5189752d1eaf444dfd6bff399feb0443977 32 32 # Puzzle 33
main.exe 10000000000 1ffffffffff d1562eb37357f9e6fc41cb2359f4d3eda4032329 32 32 # Puzzle 41
main.exe 1000000000000000000 1ffffffffffffffffff 105b7f253f0ebd7843adaebbd805c944bfb863e4 32 32 # Puzzle 73
main.exe 1000000000000000000000000000000000000000 1fffffffffffffffffffffffffffffffffffffff 242d790e5a168043c76f0539fd894b73ee67b3b3 32 32 # Puzzle 157
main.exe 10000000000000000000 1fffffffffffffffffff 783c138ac81f6a52398564bb17455576e8525b29 32 32 # Puzzle 77
```

> ⚠️ **Only use target hashes that start with a private key range of `1`.**

---

## ⚙️ Compilation

To compile the CUDA source:

```
nvcc -o main main.cu
```

---

## 🧠 Algorithm Explanation

This tool applies a sequence of **bitwise transformations** on randomly generated numbers within the specified range. These transformations include:

- Binary conversion of the number (e.g. `110011`)
- Full cycle left bit shifts
- For each shift, a vertical hex cycle from `0` to `F`
- Binary reversal
- Bit inversion

All these operations are **deeply nested** and performed **in parallel** across thousands of GPU threads. As a result:

- Each base number generates a large set of unique keys
- There are no duplicate keys in a single transformation cycle
- Extremely low probability of collisions unless base numbers are reused

---

## 🧪 Proof of Concept

Try it on **short puzzles** (e.g., Puzzle 21) to verify correctness and speed. On larger ranges, the search will naturally take longer — but this offers a unique and parallel approach to discovering keys.

---

## 💻 CPU Version

The CPU (Python) version is available here:  
👉 https://github.com/puzzleman22/Bitcoin-puzzle-transformations-CPU

---

## 📬 Contact

For questions and discussions:  
📨 [Telegram – @nmn5436](https://t.me/nmn5436)

---

## 🙏 Support

If you like this project or find a key, consider donating:

**BTC Address:**  
`bc1p6fmhpep0wkqkzvw86fg0afz85fyw692vmcg05460yuqstva7qscs2d9xhk`
