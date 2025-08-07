# The NŌSFERA Codex

## 1. Core Data Schema

All knowledge within NŌSFERA is represented by a set of interconnected JSON objects.

### 1.1 Paper Object
Represents a single research paper from the Principia Automatica.

```json
{
  "id": "PA-01",
  "title": "On the Spectral Geometry of Discourse Manifolds",
  "type": "paper",
  "status": "planned",
  "abstract": "...",
  "dependencies": [], // List of other Paper/Concept IDs it builds on
  "position": { "plane": "Plane of the Forge", "x": 100, "y": 50, "z": -200 }
}

{
  "id": "C-001",
  "name": "Graph Laplacian",
  "type": "concept",
  "description": "A matrix representation of a graph that...",
  "dependencies": ["C-002"],
  "position": { "plane": "Plane of the Forge", "x": 110, "y": 65, "z": -215 }
}

## 2. WebAssembly API Contract (v0.1)

The `libCognito.wasm` module will expose the following functions to the JavaScript environment.

### 2.1 `get_version()`
- **Description:** A simple test function to verify the WASM module is loaded and callable.
- **JS Signature:** `Module.get_version() -> string`
- **C++ Signature:** `std::string get_version()`
- **Returns:** The current version string of libCognito (e.g., "0.1.0").

### 2.2 `compute_principal_eigenvector(vectors)`
- **Description:** Takes a set of conceptual vectors, constructs the Hamiltonian matrix based on cosine similarity, and computes the principal eigenvector via power iteration. This is the "Soul-Hammer."
- **JS Signature:** `Module.compute_principal_eigenvector(js_vectors: Float32Array, num_vectors: int, vector_dim: int) -> Float32Array`
- **C++ Signature:** `emscripten::val compute_principal_eigenvector(uintptr_t js_vectors_ptr, int num_vectors, int vector_dim)`
- **Returns:** A `Float32Array` containing the principal eigenvector.