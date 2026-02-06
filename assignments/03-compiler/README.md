# Assignment 3: Bytecode Compiler and Stack VM

## Overview

Implement a compiler for a simple imperative language that compiles to bytecode, plus a stack-based virtual machine to execute it.

## Language Specification

```
program    := statement*
statement  := let_stmt | if_stmt | while_stmt | print_stmt | fn_decl | return_stmt | expr_stmt
let_stmt   := "let" IDENT "=" expr ";"
if_stmt    := "if" expr block ("else" block)?
while_stmt := "while" expr block
print_stmt := "print" expr ";"
fn_decl    := "fn" IDENT "(" params? ")" block
return_stmt:= "return" expr? ";"
expr_stmt  := expr ";"
block      := "{" statement* "}"
expr       := assignment
assignment := IDENT "=" expr | logic_or
logic_or   := logic_and ("or" logic_and)*
logic_and  := comparison ("and" comparison)*
comparison := addition (("==" | "!=" | "<" | ">" | "<=" | ">=") addition)*
addition   := multiply (("+" | "-") multiply)*
multiply   := unary (("*" | "/" | "%") unary)*
unary      := ("-" | "not") unary | call
call       := primary ("(" args? ")")*
primary    := NUMBER | STRING | "true" | "false" | "nil" | IDENT | "(" expr ")"
```

## Requirements

### Lexer
File: `src/lexer.py` — Tokenize source into token stream.

### Parser
File: `src/parser.py` — Parse tokens into AST nodes.

### Compiler
File: `src/compiler.py` — Compile AST to bytecode instructions.

### VM
File: `src/vm.py` — Execute bytecode on a stack machine.

### Convenience
File: `src/lang.py` — `execute(source: str) -> list` that runs source and returns printed output.

### Bytecode Operations (minimum)

CONST, POP, ADD, SUB, MUL, DIV, MOD, NEG, NOT, EQ, NE, LT, GT, LE, GE,
AND, OR, LOAD, STORE, JUMP, JUMP_IF_FALSE, CALL, RETURN, PRINT, HALT

## Constraints

- No eval() or exec() — implement real compilation.
- Functions have their own scope but can read outer variables.
- Integer and float arithmetic, string concatenation with +.
- Division by zero should raise a runtime error.

## Test Expectations

Tests in `tests/test_compiler.py` should cover:
- Arithmetic expressions and operator precedence
- Variable binding and scoping
- Control flow: if/else, while loops, nested blocks
- Functions: declaration, calls, recursion (fibonacci, factorial)
- String operations
- Boolean logic
- Error cases: undefined variable, division by zero, type errors
- Complex programs: FizzBuzz, bubble sort, GCD
