# Principia Semantica

**A Rigorous Framework for the Spectral Geometry of Meaning**

---

## Abstract

The dominant paradigms in topic modeling fundamentally treat corpora as collections of documents, largely ignoring the intrinsic geometric structures that underpin coherent discourse. This project challenges that view by introducing and formalizing the **Discourse Manifold Hypothesis**: the proposition that a coherent corpus forms a low-dimensional Riemannian manifold within a high-dimensional semantic space, where the manifold's geometry encodes its complete thematic structure.

_Principia Semantica_ is a research program and high-performance library designed to operationalize this hypothesis. We move beyond statistical inference and local clustering to a paradigm of **geometric analysis**. The library provides tools to:

1.  Construct a graph-based **Geometric Scaffold** of the discourse manifold.
2.  Perform **Hierarchical Spectral Thematic Decomposition** by analyzing the spectrum of the Graph Laplacian.
3.  Analyze the manifold's structure through **Geometric Invariant Interpretation**, including geodesic paths and local curvature.

This work lays the foundation for a new subfield of **Computational Manifold Semantics (CMS)**.

## The Three Pillars of CMS

1.  **Geometric Scaffolding & Laplacian Convergence (The Foundation):** Formalizes the approximation of the continuous manifold (M) with a discrete graph (G) and proves the conditions under which the Graph Laplacian (L_G) converges to the continuous Laplace-Beltrami operator (Δ_M).
2.  **Hierarchical Spectral Thematic Decomposition (The Engine):** Reframes topic discovery as finding the "eigen-themes" or "conceptual harmonics" of the manifold via the Laplacian spectrum.
3.  **Geometric Invariant Interpretation (The Telescope):** A novel framework for analyzing the structure of knowledge by computing and interpreting geodesic distances (conceptual paths) and local curvature (conceptual density).

## Installation & Usage

This project uses Docker for complete reproducibility.

```bash
# 1. Build the Docker image
docker build -t principia-semantica:latest .

# 2. Run the container with your local code mounted
docker run -it --rm -v "$(pwd)":/app principia-semantica:latest

# You are now inside the development environment.
```
