

\newpage
\section{Future Applications: guidelines from papers ?}

\subsection{Extension to Temporal Sequences}

The sample complexity framework naturally extends to video by treating time as an additional dimension:

\begin{remarkbox}[title=3D Video Transport]
For video sequences $V(x,y,t) \in \RR^{H \times W \times T \times 3}$:
\begin{itemize}[leftmargin=*]
    \item \textbf{Spatio-temporal measures}: Define $\alpha = \sum_{i,j,t} V(i,j,t) \cdot \delta_{(x_i, y_j, t)}$
    \item \textbf{Temporal coherence}: Add smoothness prior $\lambda \sum_{t} W_\varepsilon(\alpha_t, \alpha_{t+1})$
    \item \textbf{Computational cost}: Sinkhorn complexity scales as $O(K \cdot (HWT)^2)$, requiring multi-scale strategies
\end{itemize}
\end{remarkbox}

\subsection{Applications}

\begin{enumerate}[leftmargin=*]
    \item \textbf{Frame interpolation}: Compute Wasserstein barycenters between consecutive frames for smooth slow-motion effects
    \item \textbf{Video stabilization}: Transport all frames to a common reference distribution
    \item \textbf{Style transfer}: Match color distributions across video clips while preserving temporal consistency
    \item \textbf{Object tracking}: Use transport plans to infer pixel correspondences across frames
\end{enumerate}

\subsection{Volumetric Data (3D Spatial)}

For medical imaging and 3D reconstruction:

\begin{definitionbox}[title=3D Spatial Transport]
Volume data $V(x,y,z) \in \RR^{H \times W \times D}$:
\begin{itemize}[leftmargin=*]
    \item \textbf{Dimension curse}: $d=3$ yields $\varepsilon^{-3/2}$ penalty; use $\varepsilon \geq 0.05$
    \item \textbf{Sample requirement}: Need $n \sim 10^5$ voxels for $\delta = 0.01$ accuracy
    \item \textbf{Multi-scale pyramid}: Coarse-to-fine approach reduces complexity
\end{itemize}
\end{definitionbox}

\subsection{Practical Recommendations for Videos}

\begin{remarkbox}[title=Video Processing Guidelines]
\begin{itemize}[leftmargin=*]
    \item \textbf{Frame interpolation}: $\varepsilon = 0.02$, $\rho = 0.01$ (allow lighting changes)
    \item \textbf{Stabilization}: $\varepsilon = 0.05$, $\rho = 0.1$ (preserve structure)
    \item \textbf{3D volumes}: $\varepsilon = 0.05$, multi-scale pyramid with factor 2-4
    \item \textbf{Real-time processing}: $\varepsilon = 0.1$, downsample to $64 \times 64$ resolution
\end{itemize}
\end{remarkbox}

\subsection{Computational Complexity for Videos}

For video sequences with $T$ frames of size $H \times W$:
\begin{itemize}[leftmargin=*]
    \item \textbf{Naive approach}: $O(T \cdot K \cdot (HW)^2)$ per frame pair
    \item \textbf{Multi-scale}: $O(T \cdot K \cdot HW \log(HW))$ using pyramid scheme
    \item \textbf{GPU acceleration}: $50$-$100\times$ speedup with \texttt{geomloss}
    \item \textbf{Memory budget}: $O((HW)^2)$ for cost matrix; use online backend for large frames
\end{itemize}

\section{Conclusion}

We have established rigorous sample complexity bounds for Sinkhorn divergences, showing $O(n^{-1/2})$ convergence with explicit exponential dependence on regularization parameter $\varepsilon$ and polynomial dependence on dimension $d$. These theoretical results provide actionable guidelines for parameter selection in practice.

The RGB image transport experiments validate our theoretical predictions: the optimal $\varepsilon \approx 0.01$ matches the criterion derived from balancing sample and approximation errors. The Wasserstein barycenter framework extends naturally to 3D video processing, with applications in frame interpolation, stabilization, and volumetric medical imaging.

\begin{remarkbox}[title=Key Contributions]
\begin{itemize}[leftmargin=*]
    \item \textbf{Theoretical}: Sample complexity bounds connecting RKHS theory, PAC learning, and optimal transport
    \item \textbf{Practical}: Theory-driven parameter selection guidelines validated experimentally
    \item \textbf{Computational}: Multi-scale strategies for 3D video with $O(HWT \log(HWT))$ complexity
    \item \textbf{Extensions}: Framework for spatio-temporal transport in videos and volumetric data
\end{itemize}
\end{remarkbox}

Future work will focus on adaptive parameter selection, online algorithms for streaming video, and extension to manifold-valued data (e.g., orientations in 3D).
