# Aevum Agent

## Project Structure
```
aevum/
├── src/          # Core agent source code
├── tools/        # Tool definitions and implementations
├── tests/        # Test files
└── CLAUDE.md     # This file
```

## Rules
- Never read `.env` files or `.env.*` variants
- All source files should end with a newline (return)
- Tools live in `tools/`, agent logic in `src/`
