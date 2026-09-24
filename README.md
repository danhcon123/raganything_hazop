# Evaluation RAG-Anything in modular-HAZOP
## Ziele der Vorstudie

Eignung von RAG Frameworks für die Durchführung der Arbeit evaluieren

Technik und Performance test von RAG Anything für verschiedene inputfomate (Datenblätter, DEXPI files, Versuchbeschreibungen)

Kriterien:
- Kann ein graph aus jedem getestetem Input Format erzeugt werden
- Representiert der graph die im Input Format gegebenen Informationen hinreichend (todo: definiere hinreichend)
- Kann das Framework auf mehrere einzelne Graphen aus dem Framework paralel zugreifen oder diese kombinieren um komplexere aussagen zu treffen.
- Können die erzeugten teilgraphen und Ergebnisse gespeichert und wieder geladen werden.

## Prerequiresition
- uv
- ollama
- ollama's either text embeddings model or vision embeddings models / Open AI API

## Instruction
Install the program
```bash
# From root
uv sync
```

Run the program
```python
# From root
uv run python -m raganything_hazop.main
```